#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

// Forward pass kernel
__global__ void conv2d_forward_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    int output_height,
    int output_width)
{
    int n = blockIdx.z;     // batch index
    int c_out = blockIdx.y; // output channel index
    int h_out = blockIdx.x * blockDim.y + threadIdx.y;
    int w_out = threadIdx.x;

    if (h_out >= output_height || w_out >= output_width)
        return;

    float sum = bias ? bias[c_out] : 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in)
    {
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            for (int kw = 0; kw < kernel_width; ++kw)
            {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width)
                {
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width +
                                    h_in * input_width + w_in;

                    int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                     c_in * kernel_height * kernel_width +
                                     kh * kernel_width + kw;

                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = n * out_channels * output_height * output_width +
                     c_out * output_height * output_width +
                     h_out * output_width + w_out;

    output[output_idx] = sum;
}

// Backward pass kernel for input gradient
__global__ void conv2d_grad_input_kernel(
    const float *__restrict__ grad_output,
    const float *__restrict__ weight,
    float *__restrict__ grad_input,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    int output_height,
    int output_width)
{
    // Each thread computes one element of grad_input
    int w_in = blockIdx.x * blockDim.x + threadIdx.x;
    int h_in = blockIdx.y * blockDim.y + threadIdx.y;
    int c_in = blockIdx.z % in_channels;
    int n = blockIdx.z / in_channels;

    if (h_in >= input_height || w_in >= input_width)
        return;

    float sum = 0.0f;

    for (int c_out = 0; c_out < out_channels; ++c_out)
    {
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            for (int kw = 0; kw < kernel_width; ++kw)
            {
                int h_out_unpadded = h_in + padding - kh;
                int w_out_unpadded = w_in + padding - kw;

                if (h_out_unpadded % stride == 0 && w_out_unpadded % stride == 0)
                {
                    int h_out = h_out_unpadded / stride;
                    int w_out = w_out_unpadded / stride;

                    if (h_out >= 0 && h_out < output_height && w_out >= 0 && w_out < output_width)
                    {
                        int grad_output_idx = n * out_channels * output_height * output_width +
                                              c_out * output_height * output_width +
                                              h_out * output_width + w_out;

                        // Transposed convolution
                        int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                         c_in * kernel_height * kernel_width +
                                         kh * kernel_width + kw;

                        sum += grad_output[grad_output_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    int grad_input_idx = n * in_channels * input_height * input_width +
                         c_in * input_height * input_width +
                         h_in * input_width + w_in;
    grad_input[grad_input_idx] = sum;
}

// Backward pass kernel for weight and bias gradients
__global__ void conv2d_grad_weight_bias_kernel(
    const float *__restrict__ grad_output,
    const float *__restrict__ input,
    float *__restrict__ grad_weight,
    float *__restrict__ grad_bias,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    int output_height,
    int output_width)
{
    int kw = threadIdx.x;
    int kh = threadIdx.y;
    int c_in = blockIdx.y;
    int c_out = blockIdx.x;

    if (kh >= kernel_height || kw >= kernel_width)
        return;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n)
    {
        for (int h_out = 0; h_out < output_height; ++h_out)
        {
            for (int w_out = 0; w_out < output_width; ++w_out)
            {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width)
                {
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width +
                                    h_in * input_width + w_in;
                    int grad_output_idx = n * out_channels * output_height * output_width +
                                          c_out * output_height * output_width +
                                          h_out * output_width + w_out;
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }

    int grad_weight_idx = c_out * in_channels * kernel_height * kernel_width +
                          c_in * kernel_height * kernel_width +
                          kh * kernel_width + kw;

    atomicAdd(&grad_weight[grad_weight_idx], sum);

    if (grad_bias && c_in == 0 && kh == 0 && kw == 0)
    {
        float bias_sum = 0.0f;
        for (int n = 0; n < batch_size; ++n)
        {
            for (int h = 0; h < output_height; ++h)
            {
                for (int w = 0; w < output_width; ++w)
                {
                    bias_sum += grad_output[n * out_channels * output_height * output_width +
                                            c_out * output_height * output_width +
                                            h * output_width + w];
                }
            }
        }
        atomicAdd(&grad_bias[c_out], bias_sum);
    }
}

// The Autograd Function
class MyConv2dFunction : public torch::autograd::Function<MyConv2dFunction>
{
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t stride,
        int64_t padding)
    {
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
        TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
        TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor");
        if (bias.defined())
        {
            TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
            TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
        }

        // Save tensors for backward pass
        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;

        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int input_height = input.size(2);
        int input_width = input.size(3);

        int out_channels = weight.size(0);
        int kernel_height = weight.size(2);
        int kernel_width = weight.size(3);

        int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

        auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

        dim3 threads(16, 16); // 16x16 = 256 threads
        dim3 blocks((output_height + threads.y - 1) / threads.y, out_channels, batch_size);

        conv2d_forward_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding,
            output_height, output_width);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto saved_tensors = ctx->get_saved_variables();
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto bias = saved_tensors[2];

        auto stride = ctx->saved_data["stride"].toInt();
        auto padding = ctx->saved_data["padding"].toInt();

        auto grad_output = grad_outputs[0].contiguous();

        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int input_height = input.size(2);
        int input_width = input.size(3);

        int out_channels = weight.size(0);
        int kernel_height = weight.size(2);
        int kernel_width = weight.size(3);

        int output_height = grad_output.size(2);
        int output_width = grad_output.size(3);

        // Prepare output tensors for gradients
        auto grad_input = torch::zeros_like(input);
        auto grad_weight = torch::zeros_like(weight);
        auto grad_bias = bias.defined() ? torch::zeros_like(bias) : torch::Tensor();

        // --- Grad Input Calculation ---
        dim3 threads_gi(16, 16);
        dim3 blocks_gi((input_width + threads_gi.x - 1) / threads_gi.x,
                       (input_height + threads_gi.y - 1) / threads_gi.y,
                       batch_size * in_channels);

        conv2d_grad_input_kernel<<<blocks_gi, threads_gi>>>(
            grad_output.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding,
            output_height, output_width);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        // --- Grad Weight & Bias Calculation ---
        dim3 threads_gw(kernel_width, kernel_height);
        dim3 blocks_gw(out_channels, in_channels);

        conv2d_grad_weight_bias_kernel<<<blocks_gw, threads_gw>>>(
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr,
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding,
            output_height, output_width);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Main function that Python will call
torch::Tensor conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding)
{
    if (!bias.defined())
    {
        bias = torch::empty({0}, input.options());
    }
    return MyConv2dFunction::apply(input, weight, bias, stride, padding);
}
