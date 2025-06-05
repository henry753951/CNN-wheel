#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAException.h>

// 前向傳播的核心函數，使用共享記憶體來加速卷積計算
__global__ void conv2d_forward_kernel_share(
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
    extern __shared__ float s_data[]; // 動態分配共享記憶體，用來存輸入、權重和臨時輸出

    const int TILE_OH = blockDim.y; // 每個 block 處理的輸出高度瓦片大小
    const int TILE_OW = blockDim.x; // 每個 block 處理的輸出寬度瓦片大小

    // 檢查輸入參數，確保卷積核和步幅合理
    assert(kernel_height > 0);
    assert(kernel_width > 0);
    assert(stride > 0);

    // 計算共享記憶體中輸入瓦片的大小（考慮步幅和卷積核）
    const int SHARED_INPUT_HEIGHT = (TILE_OH - 1) * stride + kernel_height;
    const int SHARED_INPUT_WIDTH = (TILE_OW - 1) * stride + kernel_width;

    // 如果瓦片尺寸不合法，直接返回
    if (SHARED_INPUT_WIDTH <= 0 || SHARED_INPUT_HEIGHT <= 0)
    {
        return;
    }

    // 計算共享記憶體各部分的元素數
    const int SHARED_INPUT_SIZE_FLOATS = SHARED_INPUT_HEIGHT * SHARED_INPUT_WIDTH; // 輸入瓦片
    const int SHARED_WEIGHT_SIZE_FLOATS = kernel_height * kernel_width;            // 權重
    const int SHARED_OUTPUT_SIZE_FLOATS = TILE_OH * TILE_OW;                       // 臨時輸出

    // 在共享記憶體中分配三個區域：輸入、權重、輸出
    float *s_input_ptr = s_data;                                                         // 輸入數據起始位置
    float *s_weight_ptr = s_data + SHARED_INPUT_SIZE_FLOATS;                             // 權重數據起始位置
    float *s_output_ptr = s_data + SHARED_INPUT_SIZE_FLOATS + SHARED_WEIGHT_SIZE_FLOATS; // 輸出數據起始位置

    // 取得 thread 在 block 內的索引
    const int tx = threadIdx.x;                          // x 方向（寬度）
    const int ty = threadIdx.y;                          // y 方向（高度）
    const int thread_id_flat = ty * TILE_OW + tx;        // 將 2D 索引轉為 1D
    const int num_threads_per_block = TILE_OH * TILE_OW; // 每個 block 的 thread 數

    // 取得 block 在 grid 內的索引
    const int w_tile_idx = blockIdx.x;          // 寬度瓦片索引
    const int h_tile_idx = blockIdx.y;          // 高度瓦片索引
    const int n_cout_combined_idx = blockIdx.z; // 批次和輸出通道的組合索引

    // 確保輸出通道數有效
    assert(out_channels > 0);
    const int n = n_cout_combined_idx / out_channels;     // 批次索引
    const int c_out = n_cout_combined_idx % out_channels; // 輸出通道索引

    // 計算全局輸出像素的座標
    const int h_out_global = h_tile_idx * TILE_OH + ty;
    const int w_out_global = w_tile_idx * TILE_OW + tx;

    // 計算對應的輸入瓦片在全局輸入中的基底座標（考慮填充）
    const int h_in_global_base_for_s_input = h_tile_idx * TILE_OH * stride - padding;
    const int w_in_global_base_for_s_input = w_tile_idx * TILE_OW * stride - padding;

    // 初始化輸出和（如果有偏置，則從偏置開始）
    float sum = 0.0f;
    if (bias)
    {
        assert(c_out >= 0 && c_out < out_channels); // 確保偏置索引有效
        sum = bias[c_out];
    }

    // 遍歷所有輸入通道
    for (int c_in = 0; c_in < in_channels; ++c_in)
    {
        // 步驟 1：協同載入輸入瓦片到共享記憶體
        for (int i = thread_id_flat; i < SHARED_INPUT_SIZE_FLOATS; i += num_threads_per_block)
        {
            assert(SHARED_INPUT_WIDTH > 0);   // 確保寬度有效
            int s_h = i / SHARED_INPUT_WIDTH; // 共享記憶體中的高度索引
            int s_w = i % SHARED_INPUT_WIDTH; // 共享記憶體中的寬度索引

            // 計算共享記憶體寫入索引
            int s_idx = s_h * SHARED_INPUT_WIDTH + s_w;
            assert(s_idx >= 0 && s_idx < SHARED_INPUT_SIZE_FLOATS);

            // 計算對應的全局輸入座標
            int current_h_in_global = h_in_global_base_for_s_input + s_h;
            int current_w_in_global = w_in_global_base_for_s_input + s_w;

            // 如果在有效範圍內，從全局記憶體讀取數據；否則填 0（填充）
            if (current_h_in_global >= 0 && current_h_in_global < input_height &&
                current_w_in_global >= 0 && current_w_in_global < input_width)
            {
                assert(n >= 0 && n < batch_size);
                assert(c_in >= 0 && c_in < in_channels);

                int input_idx = n * in_channels * input_height * input_width +
                                c_in * input_height * input_width +
                                current_h_in_global * input_width + current_w_in_global;
                s_input_ptr[s_idx] = input[input_idx];
            }
            else
            {
                s_input_ptr[s_idx] = 0.0f; // 超出邊界填 0
            }
        }

        // 步驟 2：協同載入權重到共享記憶體
        if (SHARED_WEIGHT_SIZE_FLOATS > 0)
        {
            assert(kernel_width > 0); // 確保卷積核寬度有效
            for (int i = thread_id_flat; i < SHARED_WEIGHT_SIZE_FLOATS; i += num_threads_per_block)
            {
                int kh_s = i / kernel_width; // 權重高度索引
                int kw_s = i % kernel_width; // 權重寬度索引

                // 計算共享記憶體寫入索引
                int s_w_idx = kh_s * kernel_width + kw_s;
                assert(s_w_idx >= 0 && s_w_idx < SHARED_WEIGHT_SIZE_FLOATS);

                // 確保權重索引有效
                assert(c_out >= 0 && c_out < out_channels);
                assert(c_in >= 0 && c_in < in_channels);
                assert(kh_s >= 0 && kh_s < kernel_height);
                assert(kw_s >= 0 && kw_s < kernel_width);

                // 計算全局權重索引並載入
                int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                 c_in * kernel_height * kernel_width +
                                 kh_s * kernel_width + kw_s;
                s_weight_ptr[s_w_idx] = weight[weight_idx];
            }
        }
        __syncthreads(); // 同步，確保所有 thread 完成數據載入

        // 步驟 3：計算卷積
        if (h_out_global < output_height && w_out_global < output_width)
        {
            for (int kh = 0; kh < kernel_height; ++kh)
            {
                for (int kw = 0; kw < kernel_width; ++kw)
                {
                    // 計算共享記憶體中的輸入索引
                    int h_in_s = ty * stride + kh;
                    int w_in_s = tx * stride + kw;

                    // 確保共享記憶體讀取索引有效
                    assert(h_in_s >= 0 && h_in_s < SHARED_INPUT_HEIGHT);
                    assert(w_in_s >= 0 && w_in_s < SHARED_INPUT_WIDTH);
                    int s_input_idx = h_in_s * SHARED_INPUT_WIDTH + w_in_s;
                    assert(s_input_idx >= 0 && s_input_idx < SHARED_INPUT_SIZE_FLOATS);

                    // 確保權重索引有效
                    assert(kh >= 0 && kh < kernel_height);
                    assert(kw >= 0 && kw < kernel_width);
                    int s_weight_idx = kh * kernel_width + kw;
                    assert(s_weight_idx >= 0 && s_weight_idx < SHARED_WEIGHT_SIZE_FLOATS);

                    // 累加卷積結果
                    sum += s_input_ptr[s_input_idx] * s_weight_ptr[s_weight_idx];
                }
            }
        }
        __syncthreads(); // 同步，準備寫入輸出
    }

    // 步驟 4：將結果存到共享記憶體的臨時輸出
    if (h_out_global < output_height && w_out_global < output_width)
    {
        int s_out_idx = ty * TILE_OW + tx;
        assert(s_out_idx >= 0 && s_out_idx < SHARED_OUTPUT_SIZE_FLOATS);
        s_output_ptr[s_out_idx] = sum;
    }
    __syncthreads(); // 同步，確保所有 thread 完成寫入

    // 步驟 5：將共享記憶體的結果寫回全局輸出
    if (h_out_global < output_height && w_out_global < output_width)
    {
        assert(n >= 0 && n < batch_size);
        assert(c_out >= 0 && c_out < out_channels);

        // 計算全局輸出索引
        int output_idx = n * out_channels * output_height * output_width +
                         c_out * output_height * output_width +
                         h_out_global * output_width + w_out_global;

        int s_out_idx = ty * TILE_OW + tx; // 共享記憶體的輸出索引
        assert(s_out_idx >= 0 && s_out_idx < SHARED_OUTPUT_SIZE_FLOATS);
        output[output_idx] = s_output_ptr[s_out_idx];
    }
}

// 反向傳播核心：計算輸入梯度
__global__ void conv2d_grad_input_kernel_share(
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
    // 計算全局輸入索引
    int w_in = blockIdx.x * blockDim.x + threadIdx.x; // 輸入寬度索引
    int h_in = blockIdx.y * blockDim.y + threadIdx.y; // 輸入高度索引
    int c_in = blockIdx.z % in_channels;              // 輸入通道索引
    int n = blockIdx.z / in_channels;                 // 批次索引

    // 如果超出輸入範圍，直接返回
    if (h_in >= input_height || w_in >= input_width)
        return;

    float sum = 0.0f; // 累加梯度

    // 遍歷所有輸出通道和卷積核
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
__global__ void conv2d_grad_weight_bias_kernel_share(
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
class MyConv2dFunctionShare : public torch::autograd::Function<MyConv2dFunctionShare>
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
        const int TILE_OW = 16; // 輸出寬度瓦片大小
        const int TILE_OH = 16; // 輸出高度瓦片大小
        dim3 threads(TILE_OW, TILE_OH);

        // 計算 block 數量
        dim3 blocks(
            (output_width + TILE_OW - 1) / TILE_OW,  // 寬度瓦片數
            (output_height + TILE_OH - 1) / TILE_OH, // 高度瓦片數
            batch_size * out_channels                // 批次和輸出通道組合
        );

        // 計算共享記憶體大小
        const int SHARED_INPUT_HEIGHT_host = (TILE_OH - 1) * stride + kernel_height;
        const int SHARED_INPUT_WIDTH_host = (TILE_OW - 1) * stride + kernel_width;
        const int SHARED_INPUT_SIZE_FLOATS_host = SHARED_INPUT_HEIGHT_host * SHARED_INPUT_WIDTH_host;
        const int SHARED_WEIGHT_SIZE_FLOATS_host = kernel_height * kernel_width;
        const int SHARED_OUTPUT_SIZE_FLOATS_host = TILE_OH * TILE_OW;

        size_t total_shared_floats = SHARED_INPUT_SIZE_FLOATS_host +
                                     SHARED_WEIGHT_SIZE_FLOATS_host +
                                     SHARED_OUTPUT_SIZE_FLOATS_host;
        size_t shared_memory_in_bytes = total_shared_floats * sizeof(float);

        // 啟動 CUDA 核心
        conv2d_forward_kernel_share<<<blocks, threads, shared_memory_in_bytes>>>(
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

        // 計算權重和偏置梯度
        dim3 threads_gw(kernel_width, kernel_height); // 卷積核大小的 thread 配置
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

        // 返回梯度：輸入、權重、偏置、步幅（無梯度）、填充（無梯度）
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Python 調用的主函數
torch::Tensor conv2d_share(
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
    return MyConv2dFunctionShare::apply(input, weight, bias, stride, padding);
}