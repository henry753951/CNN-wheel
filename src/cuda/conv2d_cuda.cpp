#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output,
                              int input_height, int input_width, int kernel_height, int kernel_width,
                              int output_height, int output_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int ix = x + kx - kernel_width / 2;
                int iy = y + ky - kernel_height / 2;
                if (ix >= 0 && ix < input_width && iy >= 0 && iy < input_height) {
                    sum += input[iy * input_width + ix] * kernel[ky * kernel_width + kx];
                }
            }
        }
        output[y * output_width + x] = sum;
    }
}

torch::Tensor conv2d_cuda(const torch::Tensor& input, const torch::Tensor& kernel) {
    int input_height = input.size(0);
    int input_width = input.size(1);
    int kernel_height = kernel.size(0);
    int kernel_width = kernel.size(1);
    int output_height = input_height - kernel_height + 1;
    int output_width = input_width - kernel_width + 1;

    torch::Tensor output = torch::zeros({output_height, output_width}, 
                                        torch::dtype(torch::kFloat32).device(torch::kCUDA));

    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, 
                 (output_height + blockDim.y - 1) / blockDim.y);

    conv2d_kernel<<<gridDim, blockDim>>>(input.data_ptr<float>(), 
                                         kernel.data_ptr<float>(), 
                                         output.data_ptr<float>(),
                                         input_height, input_width, 
                                         kernel_height, kernel_width, 
                                         output_height, output_width);

    cudaDeviceSynchronize(); // 確保 CUDA 核函數執行完畢
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d_cuda, "2D Convolution (CUDA)");
}