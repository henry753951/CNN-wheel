#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

// 前向傳播核心：直接從全局記憶體讀取數據進行卷積計算
__global__ void conv2d_forward_kernel(
    const float *__restrict__ input,  // 輸入張量
    const float *__restrict__ weight, // 權重張量
    const float *__restrict__ bias,   // 偏置張量（可能為空）
    float *__restrict__ output,       // 輸出張量
    int batch_size,                   // 批次大小
    int in_channels,                  // 輸入通道數
    int out_channels,                 // 輸出通道數
    int input_height,                 // 輸入高度
    int input_width,                  // 輸入寬度
    int kernel_height,                // 卷積核高度
    int kernel_width,                 // 卷積核寬度
    int stride,                       // 步幅
    int padding,                      // 填充
    int output_height,                // 輸出高度
    int output_width)                 // 輸出寬度
{
    // 取得全局索引
    int n = blockIdx.z;                                // 批次索引
    int c_out = blockIdx.y;                            // 輸出通道索引
    int h_out = blockIdx.x * blockDim.y + threadIdx.y; // 輸出高度索引
    int w_out = threadIdx.x;                           // 輸出寬度索引

    // 如果輸出座標超出範圍，直接返回
    if (h_out >= output_height || w_out >= output_width)
        return;

    // 初始化輸出和（如果有偏置，從偏置開始）
    float sum = bias ? bias[c_out] : 0.0f;

    // 遍歷輸入通道和卷積核
    for (int c_in = 0; c_in < in_channels; ++c_in)
    {
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            for (int kw = 0; kw < kernel_width; ++kw)
            {
                // 計算對應的輸入座標（考慮步幅和填充）
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                // 確保輸入座標有效
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width)
                {
                    // 計算輸入和權重索引
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width +
                                    h_in * input_width + w_in;

                    int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                     c_in * kernel_height * kernel_width +
                                     kh * kernel_width + kw;

                    // 累加卷積結果
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // 寫回輸出張量
    int output_idx = n * out_channels * output_height * output_width +
                     c_out * output_height * output_width +
                     h_out * output_width + w_out;
    output[output_idx] = sum;
}

// 反向傳播核心：計算輸入梯度
__global__ void conv2d_grad_input_kernel(
    const float *__restrict__ grad_output, // 輸出梯度
    const float *__restrict__ weight,      // 權重
    float *__restrict__ grad_input,        // 輸入梯度（輸出）
    int batch_size,                        // 批次大小
    int in_channels,                       // 輸入通道數
    int out_channels,                      // 輸出通道數
    int input_height,                      // 輸入高度
    int input_width,                       // 輸入寬度
    int kernel_height,                     // 卷積核高度
    int kernel_width,                      // 卷積核寬度
    int stride,                            // 步幅
    int padding,                           // 填充
    int output_height,                     // 輸出高度
    int output_width)                      // 輸出寬度
{
    // 每個 thread 計算 grad_input 的一個元素
    int w_in = blockIdx.x * blockDim.x + threadIdx.x; // 輸入寬度索引
    int h_in = blockIdx.y * blockDim.y + threadIdx.y; // 輸入高度索引
    int c_in = blockIdx.z % in_channels;              // 輸入通道索引
    int n = blockIdx.z / in_channels;                 // 批次索引

    // 如果輸入座標超出範圍，直接返回
    if (h_in >= input_height || w_in >= input_width)
        return;

    float sum = 0.0f; // 累加梯度

    // 遍歷輸出通道和卷積核
    for (int c_out = 0; c_out < out_channels; ++c_out)
    {
        for (int kh = 0; kh < kernel_height; ++kh)
        {
            for (int kw = 0; kw < kernel_width; ++kw)
            {
                // 計算對應的輸出座標（考慮填充和步幅）
                int h_out_unpadded = h_in + padding - kh;
                int w_out_unpadded = w_in + padding - kw;

                // 檢查是否符合步幅要求
                if (h_out_unpadded % stride == 0 && w_out_unpadded % stride == 0)
                {
                    int h_out = h_out_unpadded / stride;
                    int w_out = w_out_unpadded / stride;

                    // 確保輸出座標有效
                    if (h_out >= 0 && h_out < output_height && w_out >= 0 && w_out < output_width)
                    {
                        // 計算輸出梯度和權重索引
                        int grad_output_idx = n * out_channels * output_height * output_width +
                                              c_out * output_height * output_width +
                                              h_out * output_width + w_out;

                        // 轉置卷積計算
                        int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                         c_in * kernel_height * kernel_width +
                                         kh * kernel_width + kw;

                        // 累加梯度
                        sum += grad_output[grad_output_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }

    // 寫回輸入梯度
    int grad_input_idx = n * in_channels * input_height * input_width +
                         c_in * input_height * input_width +
                         h_in * input_width + w_in;
    grad_input[grad_input_idx] = sum;
}

// 反向傳播核心：計算權重和偏置梯度
__global__ void conv2d_grad_weight_bias_kernel(
    const float *__restrict__ grad_output, // 輸出梯度
    const float *__restrict__ input,       // 輸入張量
    float *__restrict__ grad_weight,       // 權重梯度（輸出）
    float *__restrict__ grad_bias,         // 偏置梯度（輸出，可能為空）
    int batch_size,                        // 批次大小
    int in_channels,                       // 輸入通道數
    int out_channels,                      // 輸出通道數
    int input_height,                      // 輸入高度
    int input_width,                       // 輸入寬度
    int kernel_height,                     // 卷積核高度
    int kernel_width,                      // 卷積核寬度
    int stride,                            // 步幅
    int padding,                           // 填充
    int output_height,                     // 輸出高度
    int output_width)                      // 輸出寬度
{
    // 取得 thread 和 block 索引
    int kw = threadIdx.x;   // 卷積核寬度索引
    int kh = threadIdx.y;   // 卷積核高度索引
    int c_in = blockIdx.y;  // 輸入通道索引
    int c_out = blockIdx.x; // 輸出通道索引

    // 如果超出卷積核範圍，直接返回
    if (kh >= kernel_height || kw >= kernel_width)
        return;

    float sum = 0.0f; // 累加權重梯度

    // 遍歷批次和輸出特徵圖
    for (int n = 0; n < batch_size; ++n)
    {
        for (int h_out = 0; h_out < output_height; ++h_out)
        {
            for (int w_out = 0; w_out < output_width; ++w_out)
            {
                // 計算對應的輸入座標
                int h_in = h_out * stride - padding + kh;
                int w_in = w_out * stride - padding + kw;

                // 確保輸入座標有效
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width)
                {
                    // 計算輸入和輸出梯度索引
                    int input_idx = n * in_channels * input_height * input_width +
                                    c_in * input_height * input_width +
                                    h_in * input_width + w_in;
                    int grad_output_idx = n * out_channels * output_height * output_width +
                                          c_out * output_height * output_width +
                                          h_out * output_width + w_out;
                    // 累加權重梯度
                    sum += input[input_idx] * grad_output[grad_output_idx];
                }
            }
        }
    }

    // 寫回權重梯度（使用原子操作避免競爭）
    int grad_weight_idx = c_out * in_channels * kernel_height * kernel_width +
                          c_in * kernel_height * kernel_width +
                          kh * kernel_width + kw;
    atomicAdd(&grad_weight[grad_weight_idx], sum);

    // 計算偏置梯度（僅由特定 thread 處理）
    if (grad_bias && c_in == 0 && kh == 0 && kw == 0)
    {
        float bias_sum = 0.0f; // 累加偏置梯度
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
        // 寫回偏置梯度（使用原子操作）
        atomicAdd(&grad_bias[c_out], bias_sum);
    }
}

// 自定義 Autograd 函數，實現卷積的前向和反向傳播
class MyConv2dFunction : public torch::autograd::Function<MyConv2dFunction>
{
public:
    // 前向傳播
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,  // 輸入張量
        torch::Tensor weight, // 權重張量
        torch::Tensor bias,   // 偏置張量
        int64_t stride,       // 步幅
        int64_t padding)      // 填充
    {
        // 檢查輸入是否為 CUDA 張量且維度正確
        TORCH_CHECK(input.is_cuda(), "輸入必須是 CUDA 張量");
        TORCH_CHECK(weight.is_cuda(), "權重必須是 CUDA 張量");
        TORCH_CHECK(input.dim() == 4, "輸入必須是 4D 張量");
        TORCH_CHECK(weight.dim() == 4, "權重必須是 4D 張量");
        if (bias.defined())
        {
            TORCH_CHECK(bias.is_cuda(), "偏置必須是 CUDA 張量");
            TORCH_CHECK(bias.dim() == 1, "偏置必須是 1D 張量");
        }

        // 儲存反向傳播所需的張量和參數
        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;

        // 取得輸入張量尺寸
        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int input_height = input.size(2);
        int input_width = input.size(3);

        // 取得權重張量尺寸
        int out_channels = weight.size(0);
        int kernel_height = weight.size(2);
        int kernel_width = weight.size(3);

        // 計算輸出尺寸
        int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
        int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

        // 初始化輸出張量
        auto output = torch::zeros({batch_size, out_channels, output_height, output_width}, input.options());

        // 設定 thread 和 block 配置
        dim3 threads(16, 16); // 16x16 = 256 thread
        dim3 blocks(
            (output_height + threads.y - 1) / threads.y, // 高度瓦片數
            out_channels,                                // 輸出通道數
            batch_size                                   // 批次數
        );

        // 啟動 CUDA 核心
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

        C10_CUDA_KERNEL_LAUNCH_CHECK(); // 檢查核心啟動是否成功

        return output;
    }

    // 反向傳播
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        // 取得儲存的張量和參數
        auto saved_tensors = ctx->get_saved_variables();
        auto input = saved_tensors[0];
        auto weight = saved_tensors[1];
        auto bias = saved_tensors[2];

        auto stride = ctx->saved_data["stride"].toInt();
        auto padding = ctx->saved_data["padding"].toInt();

        auto grad_output = grad_outputs[0].contiguous(); // 確保輸出梯度連續

        // 取得張量尺寸
        int batch_size = input.size(0);
        int in_channels = input.size(1);
        int input_height = input.size(2);
        int input_width = input.size(3);

        int out_channels = weight.size(0);
        int kernel_height = weight.size(2);
        int kernel_width = weight.size(3);

        int output_height = grad_output.size(2);
        int output_width = grad_output.size(3);

        // 初始化梯度張量
        auto grad_input = torch::zeros_like(input);
        auto grad_weight = torch::zeros_like(weight);
        auto grad_bias = bias.defined() ? torch::zeros_like(bias) : torch::Tensor();

        // 計算輸入梯度
        dim3 threads_gi(16, 16); // 16x16 thread 配置
        dim3 blocks_gi(
            (input_width + threads_gi.x - 1) / threads_gi.x,
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

        // 計算權重和偏置梯度
        dim3 threads_gw(kernel_width, kernel_height); // 卷積核大小的 thread 配置
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

        // 返回梯度：輸入、權重、偏置、步幅（無梯度）、填充（無梯度）
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Python 調用的主函數
torch::Tensor conv2d(
    torch::Tensor input,  // 輸入張量
    torch::Tensor weight, // 權重張量
    torch::Tensor bias,   // 偏置張量
    int64_t stride,       // 步幅
    int64_t padding)      // 填充
{
    // 如果偏置未定義，創建空張量
    if (!bias.defined())
    {
        bias = torch::empty({0}, input.options());
    }
    return MyConv2dFunction::apply(input, weight, bias, stride, padding);
}