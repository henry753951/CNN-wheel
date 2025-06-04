#include <torch/extension.h>
#include <vector>

// 計算輸出尺寸
std::pair<int, int> compute_output_dims(int input_height, int input_width, int kernel_height, int kernel_width, int stride, int padding)
{
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    TORCH_CHECK(output_height > 0 && output_width > 0, "Invalid output dimensions: height=", output_height, ", width=", output_width);
    return {output_height, output_width};
}

// 前向傳播 (Forward Propagation)
void conv2d_forward_cpu(
    const float *input,  // 輸入張量
    const float *weight, // 權重張量
    const float *bias,   // 偏置張量（可選）
    float *output,       // 輸出張量
    int batch_size,      // 批次大小
    int in_channels,     // 輸入通道數
    int out_channels,    // 輸出通道數
    int input_height,    // 輸入高度
    int input_width,     // 輸入寬度
    int kernel_height,   // 卷積核高度
    int kernel_width,    // 卷積核寬度
    int stride,          // 步幅
    int padding,         // 填充
    int output_height,   // 輸出高度
    int output_width)    // 輸出寬度
{
    // 遍歷每一個輸出元素並計算其值
    for (int n = 0; n < batch_size; ++n)
    {
        for (int c_out = 0; c_out < out_channels; ++c_out)
        {
            for (int h_out = 0; h_out < output_height; ++h_out)
            {
                for (int w_out = 0; w_out < output_width; ++w_out)
                {
                    // 初始化為偏置值
                    float sum = bias ? bias[c_out] : 0.0f;
                    // 對應的卷積操作
                    for (int c_in = 0; c_in < in_channels; ++c_in)
                    {
                        for (int kh = 0; kh < kernel_height; ++kh)
                        {
                            for (int kw = 0; kw < kernel_width; ++kw)
                            {
                                int h_in = h_out * stride - padding + kh;
                                int w_in = w_out * stride - padding + kw;

                                // 確保輸入座標在有效範圍內 (處理 padding)
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
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// 反向傳播 - 計算輸入梯度 (Backward - Gradient w.r.t. Input)
void conv2d_grad_input_cpu(
    const float *grad_output,
    const float *weight,
    float *grad_input,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride, int padding,
    int output_height, int output_width)
{
    for (int n = 0; n < batch_size; ++n)
    {
        for (int c_in = 0; c_in < in_channels; ++c_in)
        {
            for (int h_in = 0; h_in < input_height; ++h_in)
            {
                for (int w_in = 0; w_in < input_width; ++w_in)
                {
                    float sum = 0.0f;
                    for (int c_out = 0; c_out < out_channels; ++c_out)
                    {
                        for (int kh = 0; kh < kernel_height; ++kh)
                        {
                            for (int kw = 0; kw < kernel_width; ++kw)
                            {
                                int h_out_nom = h_in + padding - kh;
                                int w_out_nom = w_in + padding - kw;

                                // 檢查是否能被 stride 整除，以對應前向傳播的滑動窗口
                                if (h_out_nom % stride == 0 && w_out_nom % stride == 0)
                                {
                                    int h_out = h_out_nom / stride;
                                    int w_out = w_out_nom / stride;

                                    if (h_out >= 0 && h_out < output_height && w_out >= 0 && w_out < output_width)
                                    {
                                        int grad_output_idx = n * out_channels * output_height * output_width +
                                                              c_out * output_height * output_width +
                                                              h_out * output_width + w_out;

                                        // 【關鍵錯誤修復】
                                        // 反向傳播計算輸入梯度時，需要使用 180 度翻轉後的卷積核。
                                        // 這相當於用 grad_output 對 "翻轉後的權重" 進行卷積。
                                        int flipped_kh = kernel_height - 1 - kh;
                                        int flipped_kw = kernel_width - 1 - kw;
                                        int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                                         c_in * kernel_height * kernel_width +
                                                         flipped_kh * kernel_width + flipped_kw;

                                        sum += grad_output[grad_output_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                    int grad_input_idx = n * in_channels * input_height * input_width +
                                         c_in * input_height * input_width +
                                         h_in * input_width + w_in;
                    // 這裡使用 += 是更安全的做法，儘管當前邏輯是賦值。
                    // 如果未來優化（例如並行化），累加是必須的。
                    grad_input[grad_input_idx] += sum;
                }
            }
        }
    }
}

// 反向傳播 - 計算權重和偏置梯度
void conv2d_grad_weight_bias_cpu(
    const float *grad_output,
    const float *input,
    float *grad_weight,
    float *grad_bias, // 可選
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int stride, int padding,
    int output_height, int output_width)
{

    // 計算權重梯度
    for (int c_out = 0; c_out < out_channels; ++c_out)
    {
        for (int c_in = 0; c_in < in_channels; ++c_in)
        {
            for (int kh = 0; kh < kernel_height; ++kh)
            {
                for (int kw = 0; kw < kernel_width; ++kw)
                {
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
                    grad_weight[grad_weight_idx] += sum;
                }
            }
        }
    }

    if (grad_bias)
    {
        for (int c_out = 0; c_out < out_channels; ++c_out)
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
            grad_bias[c_out] += bias_sum;
        }
    }
}

class MyConv2dFunction : public torch::autograd::Function<MyConv2dFunction>
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
        // 輸入驗證
        TORCH_CHECK(input.dim() == 4, "Input must be 4D");
        TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
        TORCH_CHECK(input.is_cpu() && weight.is_cpu(), "Inputs must be CPU tensors");
        if (bias.defined())
        {
            TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
            TORCH_CHECK(bias.is_cpu(), "Bias must be a CPU tensor");
        }

        input = input.contiguous();
        weight = weight.contiguous();
        if (bias.defined())
        {
            bias = bias.contiguous();
        }

        const int batch_size = input.size(0);
        const int in_channels = input.size(1);
        const int input_height = input.size(2);
        const int input_width = input.size(3);
        const int out_channels = weight.size(0);
        const int kernel_height = weight.size(2);
        const int kernel_width = weight.size(3);

        auto [output_height, output_width] = compute_output_dims(input_height, input_width, kernel_height, kernel_width, stride, padding);

        if (bias.defined())
        {
            TORCH_CHECK(bias.size(0) == out_channels, "Bias size must match out_channels");
        }

        auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

        conv2d_forward_cpu(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding,
            output_height, output_width);

        // 保存反向傳播所需數據
        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;

        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto stride = ctx->saved_data["stride"].toInt();
        auto padding = ctx->saved_data["padding"].toInt();
        auto grad_output = grad_outputs[0].contiguous();

        // 提取尺寸
        const int batch_size = input.size(0);
        const int in_channels = input.size(1);
        const int input_height = input.size(2);
        const int input_width = input.size(3);
        const int out_channels = weight.size(0);
        const int kernel_height = weight.size(2);
        const int kernel_width = weight.size(3);
        const int output_height = grad_output.size(2);
        const int output_width = grad_output.size(3);

        auto grad_input = torch::zeros_like(input);
        auto grad_weight = torch::zeros_like(weight);
        auto grad_bias = bias.defined() ? torch::zeros_like(bias) : torch::Tensor();

        // 計算輸入梯度
        conv2d_grad_input_cpu(
            grad_output.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding,
            output_height, output_width);

        // 計算權重和偏置梯度
        conv2d_grad_weight_bias_cpu(
            grad_output.data_ptr<float>(),
            input.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr,
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            stride, padding,
            output_height, output_width);

        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding)
{
    return MyConv2dFunction::apply(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv2d", &conv2d, "A custom CPU-based Conv2d operation (for educational purposes)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}