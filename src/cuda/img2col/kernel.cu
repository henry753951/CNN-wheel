#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            C10_CUDA_CHECK(err); \
        } \
    } while (0)

__global__ void img2col_kernel(
    const float* input, float* output,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t K_h, int64_t K_w, int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding)
{
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 輸出位置索引 (h_out * W_out + w_out)
    int64_t c_k = blockIdx.y * blockDim.y + threadIdx.y; // 通道和內核索引 (c_in * K_h * K_w)

    if (n >= N || idx >= H_out * W_out || c_k >= C_in * K_h * K_w)
        return;

    int64_t h_out = idx / W_out;
    int64_t w_out = idx % W_out;
    int64_t c_in = c_k / (K_h * K_w);
    int64_t k_h = (c_k % (K_h * K_w)) / K_w;
    int64_t k_w = c_k % K_w;

    int64_t h_in = h_out * stride + k_h - padding;
    int64_t w_in = w_out * stride + k_w - padding;

    float value = 0.0f;
    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in)
    {
        value = input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in];
    }

    output[n * (H_out * W_out * C_in * K_h * K_w) + idx * (C_in * K_h * K_w) + c_k] = value;
}

__global__ void col2im_kernel(
    const float* grad_unfolded, float* grad_padded_input,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t K_h, int64_t K_w, int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding)
{
    int64_t n = blockIdx.z;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; // 輸入位置索引 (h_in * W_in + w_in)
    int64_t c_in = blockIdx.y * blockDim.y + threadIdx.y; // 輸入通道索引

    if (n >= N || idx >= H_in * W_in || c_in >= C_in)
        return;

    int64_t h_in = idx / W_in;
    int64_t w_in = idx % W_in;
    float value = 0.0f;

    // 遍歷所有可能的輸出位置和內核位置
    for (int64_t h_out = 0; h_out < H_out; ++h_out)
    {
        for (int64_t w_out = 0; w_out < W_out; ++w_out)
        {
            int64_t h_start = h_out * stride - padding;
            int64_t w_start = w_out * stride - padding;
            for (int64_t k_h = 0; k_h < K_h; ++k_h)
            {
                for (int64_t k_w = 0; k_w < K_w; ++k_w)
                {
                    int64_t h = h_start + k_h;
                    int64_t w = w_start + k_w;
                    if (h == h_in && w == w_in)
                    {
                        int64_t c_k = c_in * K_h * K_w + k_h * K_w + k_w;
                        value += grad_unfolded[n * (H_out * W_out * C_in * K_h * K_w) +
                                              (h_out * W_out + w_out) * (C_in * K_h * K_w) + c_k];
                    }
                }
            }
        }
    }

    grad_padded_input[n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + h_in * W_in + w_in] = value;
}

void img2col_kernel(
    const float* input, float* output,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t K_h, int64_t K_w, int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding, cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (H_out * W_out + blockDim.x - 1) / blockDim.x,
        (C_in * K_h * K_w + blockDim.y - 1) / blockDim.y,
        N
    );
    img2col_kernel<<<gridDim, blockDim, 0, stream>>>(input, output, N, C_in, H_in, W_in, K_h, K_w, H_out, W_out, stride, padding);
    CUDA_CHECK(cudaGetLastError());
}

void col2im_kernel(
    const float* grad_unfolded, float* grad_padded_input,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t K_h, int64_t K_w, int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding, cudaStream_t stream)
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (H_in * W_in + blockDim.x - 1) / blockDim.x,
        (C_in + blockDim.y - 1) / blockDim.y,
        N
    );
    col2im_kernel<<<gridDim, blockDim, 0, stream>>>(grad_unfolded, grad_padded_input, N, C_in, H_in, W_in, K_h, K_w, H_out, W_out, stride, padding);
    CUDA_CHECK(cudaGetLastError());
}