#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <c10/cuda/CUDAException.h>
#include "kernel.cu"

// Autograd Function
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
        // 檢查輸入
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
        TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor: [N, C_in, H_in, W_in]");
        TORCH_CHECK(weight.dim() == 4, "Weight must be a 4D tensor: [C_out, C_in, K_h, K_w]");
        if (bias.defined())
        {
            TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
            TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor: [C_out]");
            TORCH_CHECK(bias.size(0) == weight.size(0), "Bias size must match out_channels of weight.");
        }
        TORCH_CHECK(input.size(1) == weight.size(1),
                    "Input channels (", input.size(1), ") must match weight in_channels (", weight.size(1), ").");

        // 計算維度
        int64_t N = input.size(0);
        int64_t C_in = input.size(1);
        int64_t H_in = input.size(2);
        int64_t W_in = input.size(3);
        int64_t C_out = weight.size(0);
        int64_t K_h = weight.size(2);
        int64_t K_w = weight.size(3);
        int64_t H_out = (H_in + 2 * padding - K_h) / stride + 1;
        int64_t W_out = (W_in + 2 * padding - K_w) / stride + 1;

        // 填充輸入
        torch::Tensor padded_input = torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({padding, padding, padding, padding}));

        // 分配 img2col 輸出
        torch::Tensor unfolded = torch::empty({N, H_out * W_out, C_in * K_h * K_w}, input.options());

        // 調用 img2col CUDA 內核
        img2col_kernel(
            padded_input.data_ptr<float>(),
            unfolded.data_ptr<float>(),
            N, C_in, H_in + 2 * padding, W_in + 2 * padding,
            K_h, K_w, H_out, W_out, stride, padding,
            at::cuda::getCurrentCUDAStream());

        // 重塑 weight 為矩陣
        torch::Tensor weight_flat = weight.view({C_out, C_in * K_h * K_w});

        // 執行矩陣乘法
        torch::Tensor output = unfolded.matmul(weight_flat.t()).view({N, H_out, W_out, C_out});
        output = output.permute({0, 3, 1, 2}).contiguous();

        // 添加 bias
        if (bias.defined())
        {
            output += bias.view({1, C_out, 1, 1});
        }

        // 保存必要數據
        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["stride"] = torch::tensor(stride, torch::kInt64);
        ctx->saved_data["padding"] = torch::tensor(padding, torch::kInt64);

        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        torch::Tensor input = saved[0];
        torch::Tensor weight = saved[1];
        torch::Tensor bias = saved[2];
        int64_t stride = ctx->saved_data["stride"].toInt();
        int64_t padding = ctx->saved_data["padding"].toInt();
        torch::Tensor grad_output = grad_outputs[0];

        // 計算維度
        int64_t N = input.size(0);
        int64_t C_in = input.size(1);
        int64_t H_in = input.size(2);
        int64_t W_in = input.size(3);
        int64_t C_out = weight.size(0);
        int64_t K_h = weight.size(2);
        int64_t K_w = weight.size(3);
        int64_t H_out = (H_in + 2 * padding - K_h) / stride + 1;
        int64_t W_out = (W_in + 2 * padding - K_w) / stride + 1;

        // 計算 grad_bias
        torch::Tensor grad_bias;
        if (bias.defined())
        {
            grad_bias = grad_output.sum({0, 2, 3});
        }
        else
        {
            grad_bias = torch::Tensor();
        }

        // 計算 grad_weight
        torch::Tensor padded_input = torch::nn::functional::pad(input, torch::nn::functional::PadFuncOptions({padding, padding, padding, padding}));
        torch::Tensor unfolded = torch::empty({N, H_out * W_out, C_in * K_h * K_w}, input.options());
        img2col_kernel(
            padded_input.data_ptr<float>(),
            unfolded.data_ptr<float>(),
            N, C_in, H_in + 2 * padding, W_in + 2 * padding,
            K_h, K_w, H_out, W_out, stride, padding,
            at::cuda::getCurrentCUDAStream());
        torch::Tensor grad_output_flat = grad_output.permute({0, 2, 3, 1}).contiguous().view({N * H_out * W_out, C_out});
        torch::Tensor grad_weight = (grad_output_flat.t().matmul(unfolded)).view({C_out, C_in, K_h, K_w});

        // 計算 grad_input
        torch::Tensor weight_flat = weight.view({C_out, C_in * K_h * K_w});
        torch::Tensor grad_unfolded = grad_output_flat.matmul(weight_flat).view({N, H_out * W_out, C_in * K_h * K_w});
        int64_t padded_H = H_in + 2 * padding;
        int64_t padded_W = W_in + 2 * padding;
        torch::Tensor grad_padded_input = torch::zeros({N, C_in, padded_H, padded_W}, input.options());
        col2im_kernel(
            grad_unfolded.data_ptr<float>(),
            grad_padded_input.data_ptr<float>(),
            N, C_in, padded_H, padded_W,
            K_h, K_w, H_out, W_out, stride, padding,
            at::cuda::getCurrentCUDAStream());
        torch::Tensor grad_input = grad_padded_input.slice(2, padding, padded_H - padding).slice(3, padding, padded_W - padding);

        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Python 介面
torch::Tensor conv2d_custom_cuda_pybind(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding)
{
    if (!bias.defined())
    {
        bias = torch::empty({0}, input.options());
    }
    return MyConv2dFunction::apply(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("conv2d", &conv2d_custom_cuda_pybind, "Custom Direct 2D Convolution (CUDA) with Autograd",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = torch::Tensor(),
          py::arg("stride") = 1,
          py::arg("padding") = 0);
}