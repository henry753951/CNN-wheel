#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void conv2d_kernel(
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
    int h_out = threadIdx.y + blockIdx.x * blockDim.y;
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

    // Optional ReLU activation
    output[output_idx] = sum;
}

torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                     int stride, int padding)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");

    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_height = input.size(2);
    int input_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);

    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    auto output = torch::zeros({batch_size, out_channels, output_height, output_width},
                               input.options());

    dim3 threads(output_width, 8);
    dim3 blocks((output_height + threads.y - 1) / threads.y, out_channels, batch_size);

    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        input_height, input_width,
        kernel_height, kernel_width,
        stride, padding,
        output_height, output_width);

    cudaDeviceSynchronize();

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv2d", &conv2d, "Custom 2D Convolution (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}
