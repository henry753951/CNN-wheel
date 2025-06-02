#include <torch/extension.h>

__global__ void conv2d_base_kernel(
    const float *input, const float *weight, float *output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride, int padding)
{
    int n = blockIdx.z * blockDim.z + threadIdx.z;     // batch
    int c_out = blockIdx.y * blockDim.y + threadIdx.y; // output channel
    int h_out = blockIdx.x * blockDim.x + threadIdx.x; // output height
    int w_out = threadIdx.x;                           // output width (假設簡單處理)

    if (n < batch_size && c_out < out_channels && h_out < (input_height - kernel_height + 2 * padding) / stride + 1)
    {
        float sum = 0.0f;
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
                        sum += input[n * in_channels * input_height * input_width + c_in * input_height * input_width + h_in * input_width + w_in] *
                               weight[c_out * in_channels * kernel_height * kernel_width + c_in * kernel_height * kernel_width + kh * kernel_width + kw];
                    }
                }
            }
        }
        output[n * out_channels * ((input_height - kernel_height + 2 * padding) / stride + 1) * ((input_width - kernel_width + 2 * padding) / stride + 1) +
               c_out * ((input_height - kernel_height + 2 * padding) / stride + 1) * ((input_width - kernel_width + 2 * padding) / stride + 1) +
               h_out * ((input_width - kernel_width + 2 * padding) / stride + 1) + w_out] = sum;
    }
}

torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                     int stride, int padding)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height - kernel_height + 2 * padding) / stride + 1;
    int output_width = (input_width - kernel_width + 2 * padding) / stride + 1;

    torch::Tensor output = torch::zeros(
        {batch_size, out_channels, output_height, output_width},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (output_height + blockDim.x - 1) / blockDim.x,
        (out_channels + blockDim.y - 1) / blockDim.y,
        (batch_size + blockDim.z - 1) / blockDim.z);

    conv2d_base_kernel<<<gridDim, blockDim>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_height, kernel_width,
        stride, padding);

    cudaDeviceSynchronize();
    output += bias.view({1, out_channels, 1, 1});
    return output;
}
// 在 conv2d_base.cu 中
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv2d", &conv2d, "2D Convolution (CUDA Base)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0

    );
}