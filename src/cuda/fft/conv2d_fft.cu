#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

// 檢查 CUDA 錯誤
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while (0)

// 檢查 cuFFT 錯誤
#define CHECK_CUFFT_ERROR(call) do { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        throw std::runtime_error("cuFFT error: " + std::to_string(err)); \
    } \
} while (0)

// 空間域卷積內核（前向）
__global__ void conv2d_spatial_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride, int padding,
    int output_height, int output_width)
{
    int n = blockIdx.z;     // batch
    int c_out = blockIdx.y; // 輸出通道
    int h_out = threadIdx.y + blockIdx.x * blockDim.y;
    int w_out = threadIdx.x;

    if (h_out >= output_height || w_out >= output_width) return;

    float sum = bias ? bias[c_out] : 0.0f;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width + h_in * input_width + w_in;
                    int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                     c_in * kernel_height * kernel_width + kh * kernel_width + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = n * out_channels * output_height * output_width +
                     c_out * output_height * output_width + h_out * output_width + w_out;
    output[output_idx] = sum;
}

// 空間域反向傳播：計算 grad_input
__global__ void conv2d_grad_input_kernel(
    const float *__restrict__ grad_output,
    const float *__restrict__ weight,
    float *__restrict__ grad_input,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride, int padding,
    int output_height, int output_width)
{
    int n = blockIdx.z;
    int c_in = blockIdx.y;
    int h_in = threadIdx.y + blockIdx.x * blockDim.y;
    int w_in = threadIdx.x;

    if (h_in >= input_height || w_in >= input_width) return;

    float sum = 0.0f;

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int h_out = (h_in + padding - kh) / stride;
                int w_out = (w_in + padding - kw) / stride;

                if ((h_in + padding - kh) % stride == 0 && (w_in + padding - kw) % stride == 0 &&
                    h_out >= 0 && h_out < output_height && w_out >= 0 && w_out < output_width) {
                    int grad_output_idx = n * out_channels * output_height * output_width +
                                          c_out * output_height * output_width + h_out * output_width + w_out;
                    int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                     c_in * kernel_height * kernel_width +
                                     (kernel_height - 1 - kh) * kernel_width + (kernel_width - 1 - kw);
                    sum += grad_output[grad_output_idx] * weight[weight_idx];
                }
            }
        }
    }

    int grad_input_idx = n * in_channels * input_height * input_width +
                         c_in * input_height * input_width + h_in * input_width + w_in;
    grad_input[grad_input_idx] = sum;
}

// 空間域反向傳播：計算 grad_weight
__global__ void conv2d_grad_weight_kernel(
    const float *__restrict__ input,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_weight,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride, int padding,
    int output_height, int output_width)
{
    int c_out = blockIdx.z;
    int c_in = blockIdx.y;
    int kh = threadIdx.y;
    int kw = threadIdx.x;

    if (kh >= kernel_height || kw >= kernel_width) return;

    float sum = 0.0f;

    for (int n = 0; n < batch_size; ++n) {
        for (int h_out = 0; h_out < output_height; ++h_out) {
            for (int w_out = 0; w_out < output_width; ++w_out) {
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width + h_in * input_width + w_in;
                    int grad_output_idx = n * out_channels * output_height * output_width +
                                          c_out * output_height * output_width + h_out * output_width + w_out;
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }

    int grad_weight_idx = c_out * in_channels * kernel_height * kernel_width +
                          c_in * kernel_height * kernel_width + kh * kernel_width + kw;
    grad_weight[grad_weight_idx] = sum;
}

// 空間域反向傳播：計算 grad_bias
__global__ void conv2d_grad_bias_kernel(
    const float *__restrict__ grad_output,
    float *__restrict__ grad_bias,
    int batch_size, int out_channels,
    int output_height, int output_width)
{
    int c_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_out >= out_channels) return;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                int idx = n * out_channels * output_height * output_width +
                          c_out * output_height * output_width + h * output_width + w;
                sum += grad_output[idx];
            }
        }
    }
    grad_bias[c_out] = sum;
}

// 頻域點乘內核
__global__ void complex_pointwise_multiply(
    const cufftComplex *__restrict__ input_freq,
    const cufftComplex *__restrict__ kernel_freq,
    cufftComplex *__restrict__ output_freq,
    int batch_size, int in_channels, int out_channels,
    int fft_size)
{
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= fft_size) return;

    for (int c_in = 0; c_in < in_channels; ++c_in) {
        int input_idx = n * in_channels * fft_size + c_in * fft_size + idx;
        int kernel_idx = c_out * in_channels * fft_size + c_in * fft_size + idx;
        int output_idx = n * out_channels * fft_size + c_out * fft_size + idx;

        cufftComplex a = input_freq[input_idx];
        cufftComplex b = kernel_freq[kernel_idx];
        output_freq[output_idx].x += a.x * b.x - a.y * b.y;
        output_freq[output_idx].y += a.x * b.y + a.y * b.x;
    }
}

// 自訂 Autograd Function
class Conv2dFunction : public torch::autograd::Function<Conv2dFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int stride,
        int padding,
        bool use_fft)
    {
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");

        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;
        ctx->saved_data["use_fft"] = use_fft;

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

        if (!use_fft) {
            dim3 threads(output_width, 8);
            dim3 blocks((output_height + threads.y - 1) / threads.y, out_channels, batch_size);
            conv2d_spatial_kernel<<<blocks, threads>>>(
                input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, in_channels, out_channels,
                input_height, input_width, kernel_height, kernel_width,
                stride, padding, output_height, output_width);
            CHECK_CUDA_ERROR(cudaGetLastError());
        } else {
            int fft_height = input_height + kernel_height - 1;
            int fft_width = input_width + kernel_width - 1;
            size_t fft_size = fft_height * fft_width;

            cufftHandle plan_r2c, plan_c2r;
            CHECK_CUFFT_ERROR(cufftPlan2d(&plan_r2c, fft_height, fft_width, CUFFT_R2C));
            CHECK_CUFFT_ERROR(cufftPlan2d(&plan_c2r, fft_height, fft_width, CUFFT_C2R));

            float *d_input_padded, *d_kernel_padded;
            cufftComplex *d_input_freq, *d_kernel_freq, *d_output_freq;
            CHECK_CUDA_ERROR(cudaMalloc(&d_input_padded, batch_size * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_kernel_padded, out_channels * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_input_freq, batch_size * in_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_kernel_freq, out_channels * in_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_output_freq, batch_size * out_channels * fft_size * sizeof(cufftComplex)));

            CHECK_CUDA_ERROR(cudaMemset(d_input_padded, 0, batch_size * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_kernel_freq, 0, out_channels * in_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMemset(d_output_freq, 0, batch_size * out_channels * fft_size * sizeof(cufftComplex)));

            // 填充輸入
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < in_channels; ++c) {
                    CHECK_CUDA_ERROR(cudaMemcpy2D(
                        d_input_padded + (n * in_channels + c) * fft_size + padding * fft_width + padding,
                        fft_width * sizeof(float),
                        input.data_ptr<float>() + (n * in_channels + c) * input_height * input_width,
                        input_width * sizeof(float), input_width * sizeof(float), input_height,
                        cudaMemcpyDeviceToDevice));
                }
            }

            // 填充並翻轉卷積核
            auto kernel_padded = torch::zeros({out_channels * in_channels, fft_height, fft_width}, torch::kFloat32);
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            kernel_padded[c_out * in_channels + c_in][kernel_height - 1 - kh][kernel_width - 1 - kw] =
                                weight[c_out][c_in][kh][kw];
                        }
                    }
                }
            }
            CHECK_CUDA_ERROR(cudaMemcpy(d_kernel_padded, kernel_padded.data_ptr<float>(),
                                        out_channels * in_channels * fft_size * sizeof(float),
                                        cudaMemcpyHostToDevice));

            // 執行 FFT
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < in_channels; ++c) {
                    int offset = (n * in_channels + c) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecR2C(plan_r2c, d_input_padded + offset, d_input_freq + offset));
                }
            }
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int offset = (c_out * in_channels + c_in) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecR2C(plan_r2c, d_kernel_padded + offset, d_kernel_freq + offset));
                }
            }

            // 頻域點乘
            dim3 threads_fft(256);
            dim3 blocks_fft((fft_size + threads_fft.x - 1) / threads_fft.x, out_channels, batch_size);
            complex_pointwise_multiply<<<blocks_fft, threads_fft>>>(
                d_input_freq, d_kernel_freq, d_output_freq,
                batch_size, in_channels, out_channels, fft_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            // 逆 FFT
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < out_channels; ++c) {
                    int offset = (n * out_channels + c) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecC2R(plan_c2r, d_output_freq + offset, d_input_padded + offset));
                }
            }

            // 提取輸出並應用偏置
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < out_channels; ++c) {
                    auto temp = torch::zeros({fft_height, fft_width}, torch::kFloat32);
                    CHECK_CUDA_ERROR(cudaMemcpy2D(
                        temp.data_ptr<float>(), fft_width * sizeof(float),
                        d_input_padded + (n * out_channels + c) * fft_size,
                        fft_width * sizeof(float), fft_width * sizeof(float), fft_height,
                        cudaMemcpyDeviceToHost));
                    temp = temp.slice(0, padding, padding + output_height)
                               .slice(1, padding, padding + output_width) / (fft_height * fft_width);
                    if (bias.defined()) temp += bias[c];
                    output[n][c] = temp;
                }
            }

            CHECK_CUDA_ERROR(cudaFree(d_input_padded));
            CHECK_CUDA_ERROR(cudaFree(d_kernel_padded));
            CHECK_CUDA_ERROR(cudaFree(d_input_freq));
            CHECK_CUDA_ERROR(cudaFree(d_kernel_freq));
            CHECK_CUDA_ERROR(cudaFree(d_output_freq));
            CHECK_CUFFT_ERROR(cufftDestroy(plan_r2c));
            CHECK_CUFFT_ERROR(cufftDestroy(plan_c2r));
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_output)
    {
        auto saved = ctx->get_saved_variables();
        torch::Tensor input = saved[0];
        torch::Tensor weight = saved[1];
        torch::Tensor bias = saved[2];
        int stride = ctx->saved_data["stride"].toInt();
        int padding = ctx->saved_data["padding"].toInt();
        bool use_fft = ctx->saved_data["use_fft"].toBool();

        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int input_height = input.size(2);
        int input_width = input.size(3);
        int out_channels = weight.size(0);
        int kernel_height = weight.size(2);
        int kernel_width = weight.size(3);
        int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

        auto grad_input = torch::zeros_like(input);
        auto grad_weight = torch::zeros_like(weight);
        auto grad_bias = torch::zeros_like(bias);

        if (!use_fft) {
            dim3 threads_input(input_width, 8);
            dim3 blocks_input((input_height + threads_input.y - 1) / threads_input.y, in_channels, batch_size);
            conv2d_grad_input_kernel<<<blocks_input, threads_input>>>(
                grad_output[0].data_ptr<float>(), weight.data_ptr<float>(), grad_input.data_ptr<float>(),
                batch_size, in_channels, out_channels,
                input_height, input_width, kernel_height, kernel_width,
                stride, padding, output_height, output_width);
            CHECK_CUDA_ERROR(cudaGetLastError());

            dim3 threads_weight(kernel_width, kernel_height);
            dim3 blocks_weight(1, in_channels, out_channels);
            conv2d_grad_weight_kernel<<<blocks_weight, threads_weight>>>(
                input.data_ptr<float>(), grad_output[0].data_ptr<float>(), grad_weight.data_ptr<float>(),
                batch_size, in_channels, out_channels,
                input_height, input_width, kernel_height, kernel_width,
                stride, padding, output_height, output_width);
            CHECK_CUDA_ERROR(cudaGetLastError());

            dim3 threads_bias(256);
            dim3 blocks_bias((out_channels + threads_bias.x - 1) / threads_bias.x);
            conv2d_grad_bias_kernel<<<blocks_bias, threads_bias>>>(
                grad_output[0].data_ptr<float>(), grad_bias.data_ptr<float>(),
                batch_size, out_channels, output_height, output_width);
            CHECK_CUDA_ERROR(cudaGetLastError());
        } else {
            int fft_height = input_height + kernel_height - 1;
            int fft_width = input_width + kernel_width - 1;
            size_t fft_size = fft_height * fft_width;

            cufftHandle plan_r2c, plan_c2r;
            CHECK_CUFFT_ERROR(cufftPlan2d(&plan_r2c, fft_height, fft_width, CUFFT_R2C));
            CHECK_CUFFT_ERROR(cufftPlan2d(&plan_c2r, fft_height, fft_width, CUFFT_C2R));

            float *d_grad_output_padded, *d_weight_padded, *d_grad_input_padded, *d_input_padded, *d_grad_weight_padded;
            cufftComplex *d_grad_output_freq, *d_weight_freq, *d_grad_input_freq, *d_input_freq, *d_grad_weight_freq;
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_output_padded, batch_size * out_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_weight_padded, out_channels * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_input_padded, batch_size * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_input_padded, batch_size * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weight_padded, out_channels * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_output_freq, batch_size * out_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_weight_freq, out_channels * in_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_input_freq, batch_size * in_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_input_freq, batch_size * in_channels * fft_size * sizeof(cufftComplex)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_grad_weight_freq, out_channels * in_channels * fft_size * sizeof(cufftComplex)));

            CHECK_CUDA_ERROR(cudaMemset(d_grad_output_padded, 0, batch_size * out_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_weight_padded, 0, out_channels * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_grad_input_padded, 0, batch_size * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_input_padded, 0, batch_size * in_channels * fft_size * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(d_grad_weight_padded, 0, out_channels * in_channels * fft_size * sizeof(float)));

            // 填充 grad_output
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < out_channels; ++c) {
                    CHECK_CUDA_ERROR(cudaMemcpy2D(
                        d_grad_output_padded + (n * out_channels + c) * fft_size + padding * fft_width + padding,
                        fft_width * sizeof(float),
                        grad_output[0].data_ptr<float>() + (n * out_channels + c) * output_height * output_width,
                        output_width * sizeof(float), output_width * sizeof(float), output_height,
                        cudaMemcpyDeviceToDevice));
                }
            }

            // 填充輸入
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < in_channels; ++c) {
                    CHECK_CUDA_ERROR(cudaMemcpy2D(
                        d_input_padded + (n * in_channels + c) * fft_size + padding * fft_width + padding,
                        fft_width * sizeof(float),
                        input.data_ptr<float>() + (n * in_channels + c) * input_height * input_width,
                        input_width * sizeof(float), input_width * sizeof(float), input_height,
                        cudaMemcpyDeviceToDevice));
                }
            }

            // 填充並翻轉權重
            auto kernel_padded = torch::zeros({out_channels * in_channels, fft_height, fft_width}, torch::kFloat32);
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            kernel_padded[c_out * in_channels + c_in][kernel_height - 1 - kh][kernel_width - 1 - kw] =
                                weight[c_out][c_in][kh][kw];
                        }
                    }
                }
            }
            CHECK_CUDA_ERROR(cudaMemcpy(d_weight_padded, kernel_padded.data_ptr<float>(),
                                        out_channels * in_channels * fft_size * sizeof(float),
                                        cudaMemcpyHostToDevice));

            // FFT 轉換
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < out_channels; ++c) {
                    int offset = (n * out_channels + c) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecR2C(plan_r2c, d_grad_output_padded + offset, d_grad_output_freq + offset));
                }
            }
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int offset = (c_out * in_channels + c_in) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecR2C(plan_r2c, d_weight_padded + offset, d_weight_freq + offset));
                }
            }
            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < in_channels; ++c) {
                    int offset = (n * in_channels + c) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecR2C(plan_r2c, d_input_padded + offset, d_input_freq + offset));
                }
            }

            // 計算 grad_input
            dim3 threads_fft(256);
            dim3 blocks_fft((fft_size + threads_fft.x - 1) / threads_fft.x, in_channels, batch_size);
            complex_pointwise_multiply<<<blocks_fft, threads_fft>>>(
                d_grad_output_freq, d_weight_freq, d_grad_input_freq,
                batch_size, out_channels, in_channels, fft_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < in_channels; ++c) {
                    int offset = (n * in_channels + c) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecC2R(plan_c2r, d_grad_input_freq + offset, d_grad_input_padded + offset));
                }
            }

            for (int n = 0; n < batch_size; ++n) {
                for (int c = 0; c < in_channels; ++c) {
                    auto temp = torch::zeros({fft_height, fft_width}, torch::kFloat32);
                    CHECK_CUDA_ERROR(cudaMemcpy2D(
                        temp.data_ptr<float>(), fft_width * sizeof(float),
                        d_grad_input_padded + (n * in_channels + c) * fft_size,
                        fft_width * sizeof(float), fft_width * sizeof(float), fft_height,
                        cudaMemcpyDeviceToHost));
                    grad_input[n][c] = temp.slice(0, padding, padding + input_height)
                                          .slice(1, padding, padding + input_width) / (fft_height * fft_width);
                }
            }

            // 計算 grad_weight
            dim3 blocks_fft_weight((fft_size + threads_fft.x - 1) / threads_fft.x, in_channels, out_channels);
            complex_pointwise_multiply<<<blocks_fft_weight, threads_fft>>>(
                d_input_freq, d_grad_output_freq, d_grad_weight_freq,
                batch_size, in_channels, out_channels, fft_size);
            CHECK_CUDA_ERROR(cudaGetLastError());

            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    int offset = (c_out * in_channels + c_in) * fft_size;
                    CHECK_CUFFT_ERROR(cufftExecC2R(plan_c2r, d_grad_weight_freq + offset, d_grad_weight_padded + offset));
                }
            }

            auto grad_weight_padded = torch::zeros({out_channels * in_channels, fft_height, fft_width}, torch::kFloat32);
            CHECK_CUDA_ERROR(cudaMemcpy(grad_weight_padded.data_ptr<float>(), d_grad_weight_padded,
                                        out_channels * in_channels * fft_size * sizeof(float),
                                        cudaMemcpyDeviceToHost));
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int c_in = 0; c_in < in_channels; ++c_in) {
                    for (int kh = 0; kh < kernel_height; ++kh) {
                        for (int kw = 0; kw < kernel_width; ++kw) {
                            grad_weight[c_out][c_in][kh][kw] =
                                grad_weight_padded[c_out * in_channels + c_in][kh][kw] / (fft_height * fft_width);
                        }
                    }
                }
            }

            // 計算 grad_bias
            dim3 threads_bias(256);
            dim3 blocks_bias((out_channels + threads_bias.x - 1) / threads_bias.x);
            conv2d_grad_bias_kernel<<<blocks_bias, threads_bias>>>(
                grad_output[0].data_ptr<float>(), grad_bias.data_ptr<float>(),
                batch_size, out_channels, output_height, output_width);
            CHECK_CUDA_ERROR(cudaGetLastError());

            CHECK_CUDA_ERROR(cudaFree(d_grad_output_padded));
            CHECK_CUDA_ERROR(cudaFree(d_weight_padded));
            CHECK_CUDA_ERROR(cudaFree(d_grad_input_padded));
            CHECK_CUDA_ERROR(cudaFree(d_input_padded));
            CHECK_CUDA_ERROR(cudaFree(d_grad_weight_padded));
            CHECK_CUDA_ERROR(cudaFree(d_grad_output_freq));
            CHECK_CUDA_ERROR(cudaFree(d_weight_freq));
            CHECK_CUDA_ERROR(cudaFree(d_grad_input_freq));
            CHECK_CUDA_ERROR(cudaFree(d_input_freq));
            CHECK_CUDA_ERROR(cudaFree(d_grad_weight_freq));
            CHECK_CUFFT_ERROR(cufftDestroy(plan_r2c));
            CHECK_CUFFT_ERROR(cufftDestroy(plan_c2r));
        }

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

// Python 綁定
torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                     int stride, int padding, bool use_fft) {
    return Conv2dFunction::apply(input, weight, bias, stride, padding, use_fft);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "Custom 2D convolution (CUDA, FFT support)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("padding") = 0, py::arg("use_fft") = false);
}