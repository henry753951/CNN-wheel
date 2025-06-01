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