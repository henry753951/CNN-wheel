#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

// Forward pass kernel (Revised for easier debugging with DSA)
__global__ void conv2d_forward_kernel_share( // Renamed for clarity
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int batch_size,    // Host should ensure > 0
    int in_channels,   // Host should ensure > 0
    int out_channels,  // Host should ensure > 0
    int input_height,  // Host should ensure > 0
    int input_width,   // Host should ensure > 0
    int kernel_height, // Host should ensure > 0
    int kernel_width,  // Host should ensure > 0
    int stride,        // Host should ensure > 0
    int padding,       // Host should ensure >= 0
    int output_height, // Host should ensure > 0
    int output_width)  // Host should ensure > 0
{
    // Dynamic shared memory
    extern __shared__ float s_data[];

    // Tile dimensions from blockDim
    const int TILE_OH = blockDim.y;
    const int TILE_OW = blockDim.x;
    // Assert that block dimensions are valid (CUDA runtime usually catches this, but for completeness)
    // These asserts are more for documentation/understanding within the kernel.
    // Realistically, if blockDim is 0, the kernel launch itself would likely fail.
    // assert(TILE_OH > 0 && TILE_OW > 0);

    // Shared memory size calculations
    // Host MUST ensure kernel_height, kernel_width, stride are > 0
    // Assertions here will catch issues if host validation failed AND DSA is on.
    assert(kernel_height > 0);
    assert(kernel_width > 0);
    assert(stride > 0);

    const int SHARED_INPUT_HEIGHT = (TILE_OH - 1) * stride + kernel_height;
    const int SHARED_INPUT_WIDTH = (TILE_OW - 1) * stride + kernel_width;

    // Critical check for preventing division by zero and invalid shared mem array dimensions
    if (SHARED_INPUT_WIDTH <= 0 || SHARED_INPUT_HEIGHT <= 0)
    {
        // Using assert(false) can help pinpoint this if DSA is on
        // assert(!"SHARED_INPUT_WIDTH or SHARED_INPUT_HEIGHT is zero or negative!");
        return;
    }

    const int SHARED_INPUT_SIZE_FLOATS = SHARED_INPUT_HEIGHT * SHARED_INPUT_WIDTH;
    const int SHARED_WEIGHT_SIZE_FLOATS = kernel_height * kernel_width;
    const int SHARED_OUTPUT_SIZE_FLOATS = TILE_OH * TILE_OW;

    // Pointers to shared memory regions
    float *s_input_ptr = s_data;
    float *s_weight_ptr = s_data + SHARED_INPUT_SIZE_FLOATS;
    float *s_output_ptr = s_data + SHARED_INPUT_SIZE_FLOATS + SHARED_WEIGHT_SIZE_FLOATS;

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread_id_flat = ty * TILE_OW + tx;
    const int num_threads_per_block = TILE_OH * TILE_OW;

    // Global indices
    const int w_tile_idx = blockIdx.x;
    const int h_tile_idx = blockIdx.y;
    const int n_cout_combined_idx = blockIdx.z;

    // Assertions for block indices (host grid configuration should ensure this)
    // assert(w_tile_idx * TILE_OW < output_width_or_grid_dim_related_bound); // Hard to get exact bound here
    // assert(h_tile_idx * TILE_OH < output_height_or_grid_dim_related_bound);
    // assert(n_cout_combined_idx < batch_size * out_channels_or_grid_dim_z);

    assert(out_channels > 0); // Should be validated by host
    const int n = n_cout_combined_idx / out_channels;
    const int c_out = n_cout_combined_idx % out_channels;

    const int h_out_global = h_tile_idx * TILE_OH + ty;
    const int w_out_global = w_tile_idx * TILE_OW + tx;

    const int h_in_global_base_for_s_input = h_tile_idx * TILE_OH * stride - padding;
    const int w_in_global_base_for_s_input = w_tile_idx * TILE_OW * stride - padding;

    // Bias access: ensure c_out is valid for bias array if bias is not null
    float sum = 0.0f;
    if (bias)
    {
        assert(c_out >= 0 && c_out < out_channels); // Assert index before access
        sum = bias[c_out];
    }

    for (int c_in = 0; c_in < in_channels; ++c_in)
    {
        // 1. Load input tile
        for (int i = thread_id_flat; i < SHARED_INPUT_SIZE_FLOATS; i += num_threads_per_block)
        {
            assert(SHARED_INPUT_WIDTH > 0); // Already checked, but good for local reasoning
            int s_h = i / SHARED_INPUT_WIDTH;
            int s_w = i % SHARED_INPUT_WIDTH;

            // Assert shared memory write index bounds
            int s_idx = s_h * SHARED_INPUT_WIDTH + s_w;
            assert(s_idx >= 0 && s_idx < SHARED_INPUT_SIZE_FLOATS);

            int current_h_in_global = h_in_global_base_for_s_input + s_h;
            int current_w_in_global = w_in_global_base_for_s_input + s_w;

            if (current_h_in_global >= 0 && current_h_in_global < input_height &&
                current_w_in_global >= 0 && current_w_in_global < input_width)
            {
                // Assert global read index bounds (components)
                assert(n >= 0 && n < batch_size);
                assert(c_in >= 0 && c_in < in_channels);
                // current_h_in_global and current_w_in_global are already checked by the if

                int input_idx = n * in_channels * input_height * input_width +
                                c_in * input_height * input_width +
                                current_h_in_global * input_width + current_w_in_global;
                // Optional: assert(input_idx >= 0 && input_idx < total_input_elements); // total_input_elements is hard to get here
                s_input_ptr[s_idx] = input[input_idx];
            }
            else
            {
                s_input_ptr[s_idx] = 0.0f; // Apply padding
            }
        }

        // 2. Load weight tile
        if (SHARED_WEIGHT_SIZE_FLOATS > 0)
        {
            assert(kernel_width > 0); // Already checked, but good for local reasoning
            for (int i = thread_id_flat; i < SHARED_WEIGHT_SIZE_FLOATS; i += num_threads_per_block)
            {
                int kh_s = i / kernel_width;
                int kw_s = i % kernel_width;

                // Assert shared memory write index bounds
                int s_w_idx = kh_s * kernel_width + kw_s;
                assert(s_w_idx >= 0 && s_w_idx < SHARED_WEIGHT_SIZE_FLOATS);

                // Assert global read index bounds (components)
                assert(c_out >= 0 && c_out < out_channels);
                assert(c_in >= 0 && c_in < in_channels);
                assert(kh_s >= 0 && kh_s < kernel_height);
                assert(kw_s >= 0 && kw_s < kernel_width);

                int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                 c_in * kernel_height * kernel_width +
                                 kh_s * kernel_width + kw_s;
                s_weight_ptr[s_w_idx] = weight[weight_idx];
            }
        }
        __syncthreads();

        // 3. Compute convolution
        if (h_out_global < output_height && w_out_global < output_width)
        {
            for (int kh = 0; kh < kernel_height; ++kh)
            {
                for (int kw = 0; kw < kernel_width; ++kw)
                {
                    int h_in_s = ty * stride + kh;
                    int w_in_s = tx * stride + kw;

                    // Assert shared memory read index bounds
                    assert(h_in_s >= 0 && h_in_s < SHARED_INPUT_HEIGHT);
                    assert(w_in_s >= 0 && w_in_s < SHARED_INPUT_WIDTH);
                    int s_input_idx = h_in_s * SHARED_INPUT_WIDTH + w_in_s;
                    assert(s_input_idx >= 0 && s_input_idx < SHARED_INPUT_SIZE_FLOATS);

                    assert(kh >= 0 && kh < kernel_height); // From loop bounds
                    assert(kw >= 0 && kw < kernel_width);  // From loop bounds
                    int s_weight_idx = kh * kernel_width + kw;
                    assert(s_weight_idx >= 0 && s_weight_idx < SHARED_WEIGHT_SIZE_FLOATS);

                    sum += s_input_ptr[s_input_idx] * s_weight_ptr[s_weight_idx];
                }
            }
        }
        __syncthreads();
    }

    // 4. Store result to s_output_ptr
    if (h_out_global < output_height && w_out_global < output_width)
    {
        int s_out_idx = ty * TILE_OW + tx;
        assert(s_out_idx >= 0 && s_out_idx < SHARED_OUTPUT_SIZE_FLOATS);
        s_output_ptr[s_out_idx] = sum;
    }
    __syncthreads();

    // 5. Write from s_output_ptr to global output
    if (h_out_global < output_height && w_out_global < output_width)
    {
        // Assert global write index bounds (components)
        assert(n >= 0 && n < batch_size);
        assert(c_out >= 0 && c_out < out_channels);
        // h_out_global and w_out_global are checked by the if condition

        int output_idx = n * out_channels * output_height * output_width +
                         c_out * output_height * output_width +
                         h_out_global * output_width + w_out_global;

        int s_out_idx = ty * TILE_OW + tx;                               // Index for reading from s_output_ptr
        assert(s_out_idx >= 0 && s_out_idx < SHARED_OUTPUT_SIZE_FLOATS); // Should be valid from write
        output[output_idx] = s_output_ptr[s_out_idx];
    }
}

// Backward pass kernel for input gradient
__global__ void conv2d_grad_input_kernel_share(
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

                        // This is a transposed convolution, so kernel weights are flipped
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
__global__ void conv2d_grad_weight_bias_kernel_share(
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
    // Each thread block computes gradients for one filter weight
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

    // Use atomicAdd to avoid race conditions when multiple threads/blocks update the same weight gradient
    atomicAdd(&grad_weight[grad_weight_idx], sum);

    // For bias, threads in the first input channel and first kernel element compute it
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
class MyConv2dFunctionShare : public torch::autograd::Function<MyConv2dFunctionShare>
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

        const int TILE_OW = 16; // Corresponds to blockDim.x in kernel, processing output width
        const int TILE_OH = 16; // Corresponds to blockDim.y in kernel, processing output height
        dim3 threads(TILE_OW, TILE_OH);

        // Corrected block dimension calculation based on kernel's interpretation:
        // kernel: w_tile_idx = blockIdx.x; h_tile_idx = blockIdx.y; n_cout_combined_idx = blockIdx.z;
        dim3 blocks(
            (output_width + TILE_OW - 1) / TILE_OW,  // Grid dim for output width tiles (blockIdx.x)
            (output_height + TILE_OH - 1) / TILE_OH, // Grid dim for output height tiles (blockIdx.y)
            batch_size * out_channels                // Grid dim for combined batch and output_channel (blockIdx.z)
        );

        // Calculate shared memory size in bytes
        // These calculations MUST match the logic inside your CUDA kernel
        const int SHARED_INPUT_HEIGHT_host = (TILE_OH - 1) * stride + kernel_height;
        const int SHARED_INPUT_WIDTH_host = (TILE_OW - 1) * stride + kernel_width;
        const int SHARED_INPUT_SIZE_FLOATS_host = SHARED_INPUT_HEIGHT_host * SHARED_INPUT_WIDTH_host;
        const int SHARED_WEIGHT_SIZE_FLOATS_host = kernel_height * kernel_width;
        const int SHARED_OUTPUT_SIZE_FLOATS_host = TILE_OH * TILE_OW;

        size_t total_shared_floats = SHARED_INPUT_SIZE_FLOATS_host +
                                     SHARED_WEIGHT_SIZE_FLOATS_host +
                                     SHARED_OUTPUT_SIZE_FLOATS_host;
        size_t shared_memory_in_bytes = total_shared_floats * sizeof(float);

        // Launch the kernel
        // Ensure your kernel is named conv2d_forward_kernel or conv2d_forward_kernel_debuggable
        conv2d_forward_kernel_share<<<blocks, threads, shared_memory_in_bytes>>>( // Or your actual kernel name
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding, // Pass the int versions
            output_height, output_width);

        C10_CUDA_KERNEL_LAUNCH_CHECK(); // Checks for errors from the kernel launch/execution

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

        conv2d_grad_input_kernel_share<<<blocks_gi, threads_gi>>>(
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

        conv2d_grad_weight_bias_kernel_share<<<blocks_gw, threads_gw>>>(
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

        // Return gradients for input, weight, bias, stride, padding
        // Gradients for non-tensor inputs (stride, padding) are torch::Tensor()
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Main function that Python will call
torch::Tensor conv2d_share(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding)
{
    // Make sure bias is defined, even if it's empty, to pass to the function
    if (!bias.defined())
    {
        bias = torch::empty({0}, input.options());
    }
    return MyConv2dFunctionShare::apply(input, weight, bias, stride, padding);
}
