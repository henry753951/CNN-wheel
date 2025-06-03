#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

// CUDA kernel for img2col transformation
__global__ void img2col_kernel(
    const float *input,
    float *output,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    int output_height,
    int output_width)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size)
        return;

    int col_size = in_channels * kernel_height * kernel_width;
    int row_size = output_height * output_width;

    if (col_idx >= col_size * row_size)
        return;

    int out_col = col_idx % row_size; // This is the output spatial index (flattened)
    int out_row = col_idx / row_size; // This is the combined (in_channel, k_y, k_x) index

    int out_x = out_col % output_width;
    int out_y = out_col / output_width;

    int in_c = out_row / (kernel_height * kernel_width);
    int k_y = (out_row % (kernel_height * kernel_width)) / kernel_width;
    int k_x = out_row % kernel_width;

    int in_x = out_x * stride - padding + k_x;
    int in_y = out_y * stride - padding + k_y;

    float value = 0.0f;
    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height)
    {
        int input_idx = batch_idx * (in_channels * input_height * input_width) +
                        in_c * (input_height * input_width) +
                        in_y * input_width + in_x;
        value = input[input_idx];
    }

    int output_idx = batch_idx * (col_size * row_size) + col_idx;
    output[output_idx] = value;
}

// CUDA kernel for col2img transformation (for grad_input in backward pass)
__global__ void col2img_kernel(
    const float *columns,
    float *grad_input,
    int batch_size,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int padding,
    int output_height,
    int output_width)
{
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size)
        return;

    int col_size = in_channels * kernel_height * kernel_width;
    int row_size = output_height * output_width;

    if (col_idx >= col_size * row_size)
        return;

    int out_col = col_idx % row_size;
    int out_row = col_idx / row_size;

    int out_x = out_col % output_width;
    int out_y = out_col / output_width;

    int in_c = out_row / (kernel_height * kernel_width);
    int k_y = (out_row % (kernel_height * kernel_width)) / kernel_width;
    int k_x = out_row % kernel_width;

    int in_x = out_x * stride - padding + k_x;
    int in_y = out_y * stride - padding + k_y;

    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height)
    {
        int input_idx = batch_idx * (in_channels * input_height * input_width) +
                        in_c * (input_height * input_width) +
                        in_y * input_width + in_x;
        // Use atomicAdd because multiple output pixels might write to the same input pixel
        atomicAdd(&grad_input[input_idx], columns[batch_idx * (col_size * row_size) + col_idx]);
    }
}

// CUDA kernel for adding bias
__global__ void add_bias_kernel(
    float *output,
    const float *bias,
    int batch_size,
    int out_channels,
    int output_height,
    int output_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (batch_idx >= batch_size)
        return;

    int out_c = idx / (output_height * output_width);
    if (out_c >= out_channels)
        return;

    int out_idx = batch_idx * (out_channels * output_height * output_width) +
                  out_c * (output_height * output_width) +
                  (idx % (output_height * output_width));

    output[out_idx] += bias[out_c];
}

// CUDA kernel for calculating grad_weight (direct convolution of input with grad_output)
__global__ void conv2d_grad_weight_kernel(
    const float *input,       // Original input
    const float *grad_output, // Gradient from the next layer
    float *grad_weight,       // Output: gradient for weights
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
    // Each thread computes one element of grad_weight
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

    // Use atomicAdd since multiple blocks might contribute to the same weight gradient
    atomicAdd(&grad_weight[grad_weight_idx], sum);
}

// CUDA kernel for calculating grad_bias
__global__ void conv2d_grad_bias_kernel(
    const float *grad_output,
    float *grad_bias,
    int batch_size,
    int out_channels,
    int output_height,
    int output_width)
{
    int c_out = blockIdx.x; // Each block computes gradient for one output channel bias

    if (c_out >= out_channels)
        return;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n)
    {
        for (int h_out = 0; h_out < output_height; ++h_out)
        {
            for (int w_out = 0; w_out < output_width; ++w_out)
            {
                sum += grad_output[n * out_channels * output_height * output_width +
                                   c_out * output_height * output_width +
                                   h_out * output_width + w_out];
            }
        }
    }
    atomicAdd(&grad_bias[c_out], sum);
}

// The Autograd Function for img2col-based convolution
class Conv2dIm2ColFunction : public torch::autograd::Function<Conv2dIm2ColFunction>
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
        // Input validation
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
        TORCH_CHECK(input.dim() == 4, "Input must be 4D: [batch_size, in_channels, input_height, input_width]");
        TORCH_CHECK(weight.dim() == 4, "Weight must be 4D: [out_channels, in_channels, kernel_height, kernel_width]");
        if (bias.defined())
        {
            TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
            TORCH_CHECK(bias.dim() == 1, "Bias must be 1D: [out_channels]");
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
        int weight_in_channels = weight.size(1);
        int kernel_height = weight.size(2);
        int kernel_width = weight.size(3);

        // Check consistency
        TORCH_CHECK(in_channels == weight_in_channels,
                    "Input channels (", in_channels, ") must match weight in_channels (", weight_in_channels, ")");
        if (bias.defined())
        {
            TORCH_CHECK(bias.size(0) == out_channels,
                        "Bias size (", bias.size(0), ") must match out_channels (", out_channels, ")");
        }

        // Calculate output dimensions
        int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
        TORCH_CHECK(output_height > 0 && output_width > 0,
                    "Invalid output dimensions: output_height=", output_height, ", output_width=", output_width);

        auto output = torch::zeros(
            {batch_size, out_channels, output_height, output_width},
            input.options());

        // Create im2col matrix
        int col_height = in_channels * kernel_height * kernel_width;
        int col_width = output_height * output_width;
        auto columns = torch::zeros(
            {batch_size, col_height, col_width},
            input.options());

        // Set up CUDA grid and block dimensions for img2col
        dim3 blockDim_img2col(16, 16); // Example block size
        dim3 gridDim_img2col(
            (col_height * col_width + blockDim_img2col.x - 1) / blockDim_img2col.x,
            (batch_size + blockDim_img2col.y - 1) / blockDim_img2col.y);

        // Launch img2col kernel
        {
            at::cuda::CUDAGuard device_guard(input.device());
            img2col_kernel<<<gridDim_img2col, blockDim_img2col>>>(
                input.data_ptr<float>(),
                columns.data_ptr<float>(),
                batch_size,
                in_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                stride,
                padding,
                output_height,
                output_width);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // Initialize cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Perform batch matrix multiplication with cuBLAS
        float alpha = 1.0f;
        float beta = 0.0f;
        auto weight_reshaped = weight.view({out_channels, col_height}).contiguous();

        for (int b = 0; b < batch_size; ++b)
        {
            auto input_slice = columns.select(0, b).contiguous();
            auto output_slice = output.select(0, b).view({out_channels, col_width}).contiguous();

            // C = A * B
            // A: weight_reshaped (out_channels x col_height)
            // B: input_slice (col_height x col_width)
            // C: output_slice (out_channels x col_width)
            // Using CUBLAS_OP_T for transA and transB to align with PyTorch's row-major storage
            // This computes C^T = B^T * A^T
            cublasSgemm(
                handle,
                CUBLAS_OP_T,  // transa: A is B_matrix_cuBLAS = input_slice (transposed)
                CUBLAS_OP_T,  // transb: B is A_matrix_cuBLAS = weight_reshaped (transposed)
                col_width,    // m: rows of C_cuBLAS (C^T) = col_width
                out_channels, // n: columns of C_cuBLAS (C^T) = out_channels
                col_height,   // k: inner dimension = col_height
                &alpha,
                input_slice.data_ptr<float>(), col_width,      // A_cuBLAS (input_slice), lda (rows of input_slice's transpose, which is its number of columns)
                weight_reshaped.data_ptr<float>(), col_height, // B_cuBLAS (weight_reshaped), ldb (rows of weight_reshaped's transpose, which is its number of columns)
                &beta,
                output_slice.data_ptr<float>(), col_width); // C_cuBLAS (output_slice), ldc (rows of output_slice's transpose, which is its number of columns)
        }

        // Clean up cuBLAS handle
        cublasDestroy(handle);

        // Add bias if defined
        if (bias.defined())
        {
            dim3 biasBlockDim(16, 16);
            dim3 biasGridDim(
                (out_channels * output_height * output_width + biasBlockDim.x - 1) / biasBlockDim.x,
                (batch_size + biasBlockDim.y - 1) / biasBlockDim.y);

            {
                at::cuda::CUDAGuard device_guard(input.device());
                add_bias_kernel<<<biasGridDim, biasBlockDim>>>(
                    output.data_ptr<float>(),
                    bias.data_ptr<float>(),
                    batch_size,
                    out_channels,
                    output_height,
                    output_width);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }

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

        auto grad_output = grad_outputs[0].contiguous(); // Ensure contiguous for kernel access

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

        // --- Grad Input Calculation (using col2img approach) ---
        // grad_columns = grad_output_reshaped @ weight_transposed
        // grad_columns (batch_size, col_height, col_width)

        int col_height = in_channels * kernel_height * kernel_width;
        int col_width = output_height * output_width;

        auto grad_columns = torch::zeros(
            {batch_size, col_height, col_width},
            input.options());

        // Reshape grad_output for GEMM
        // grad_output_reshaped (batch_size, out_channels, col_width)
        auto grad_output_reshaped = grad_output.view({batch_size, out_channels, col_width}).contiguous();
        // Weight needs to be transposed for grad_input calculation
        // weight_transposed (col_height x out_channels)
        auto weight_transposed = weight.view({out_channels, col_height}).transpose(0, 1).contiguous();

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        for (int b = 0; b < batch_size; ++b)
        {
            auto grad_output_slice = grad_output_reshaped.select(0, b).contiguous(); // (out_channels x col_width)
            auto grad_columns_slice = grad_columns.select(0, b).contiguous();        // (col_height x col_width)

            // grad_columns_slice = weight_transposed * grad_output_slice
            // C = A * B
            // A: weight_transposed (col_height x out_channels)
            // B: grad_output_slice (out_channels x col_width)
            // C: grad_columns_slice (col_height x col_width)
            // Using CUBLAS_OP_T for transA and transB to align with PyTorch's row-major storage
            // This computes C^T = B^T * A^T
            cublasSgemm(
                handle,
                CUBLAS_OP_T,  // transa: A is B_matrix_cuBLAS = grad_output_slice (transposed)
                CUBLAS_OP_T,  // transb: B is A_matrix_cuBLAS = weight_transposed (transposed)
                col_width,    // m: rows of C_cuBLAS (C^T) = col_width
                col_height,   // n: columns of C_cuBLAS (C^T) = col_height
                out_channels, // k: inner dimension = out_channels
                &alpha,
                grad_output_slice.data_ptr<float>(), col_width,    // A_cuBLAS (grad_output_slice), lda (rows of grad_output_slice's transpose, which is its number of columns)
                weight_transposed.data_ptr<float>(), out_channels, // B_cuBLAS (weight_transposed), ldb (rows of weight_transposed's transpose, which is its number of columns)
                &beta,
                grad_columns_slice.data_ptr<float>(), col_width); // C_cuBLAS (grad_columns_slice), ldc (rows of grad_columns_slice's transpose, which is its number of columns)
        }
        cublasDestroy(handle);

        // Apply col2img to get grad_input
        dim3 blockDim_col2img(16, 16);
        dim3 gridDim_col2img(
            (col_height * col_width + blockDim_col2img.x - 1) / blockDim_col2img.x,
            (batch_size + blockDim_col2img.y - 1) / blockDim_col2img.y);

        {
            at::cuda::CUDAGuard device_guard(input.device());
            col2img_kernel<<<gridDim_col2img, blockDim_col2img>>>(
                grad_columns.data_ptr<float>(),
                grad_input.data_ptr<float>(),
                batch_size,
                in_channels,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                stride,
                padding,
                output_height,
                output_width);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // --- Grad Weight Calculation ---
        dim3 threads_gw(kernel_width, kernel_height); // Threads per kernel element
        dim3 blocks_gw(out_channels, in_channels);    // Blocks for each (out_channel, in_channel) pair

        {
            at::cuda::CUDAGuard device_guard(input.device());
            conv2d_grad_weight_kernel<<<blocks_gw, threads_gw>>>(
                input.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                batch_size, in_channels, out_channels,
                input_height, input_width,
                kernel_height, kernel_width,
                stride, padding,
                output_height, output_width);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        // --- Grad Bias Calculation ---
        if (bias.defined())
        {
            dim3 threads_gb(32); // Each block computes for one bias element
            dim3 blocks_gb(out_channels);

            {
                at::cuda::CUDAGuard device_guard(input.device());
                conv2d_grad_bias_kernel<<<blocks_gb, threads_gb>>>(
                    grad_output.data_ptr<float>(),
                    grad_bias.data_ptr<float>(),
                    batch_size,
                    out_channels,
                    output_height,
                    output_width);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }

        // Return gradients for input, weight, bias.
        // Gradients for non-tensor inputs (stride, padding) are torch::Tensor().
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Python-facing function that uses the Autograd Function
torch::Tensor conv2d_img2col_pybind(
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
    return Conv2dIm2ColFunction::apply(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv2d", &conv2d_img2col_pybind, "Im2col 2D Convolution (CUDA) with Autograd",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(), // Default to undefined tensor
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}