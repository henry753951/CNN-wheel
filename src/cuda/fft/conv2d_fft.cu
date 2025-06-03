#include <torch/extension.h>
#include <cudnn.h>
#include <stdexcept>

// 檢查 CUDA 錯誤
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while (0)

// 檢查 cuDNN 錯誤
#define CHECK_CUDNN_ERROR(call) do { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error("cuDNN error: " + std::string(cudnnGetErrorString(status))); \
    } \
} while (0)

// 自訂 Autograd Function
class Conv2dCuDNNFunction : public torch::autograd::Function<Conv2dCuDNNFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int stride,
        int padding)
    {
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
        TORCH_CHECK(bias.is_cuda() || !bias.defined(), "Bias must be a CUDA tensor or undefined");
        TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
        TORCH_CHECK(!bias.defined() || bias.dtype() == torch::kFloat32, "Bias must be float32");

        ctx->save_for_backward({input, weight, bias});
        ctx->saved_data["stride"] = stride;
        ctx->saved_data["padding"] = padding;

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

        cudnnHandle_t cudnn_handle;
        CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle));

        cudnnTensorDescriptor_t input_desc;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
            input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size, in_channels, input_height, input_width));

        cudnnFilterDescriptor_t weight_desc;
        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&weight_desc));
        CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(
            weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            out_channels, in_channels, kernel_height, kernel_width));

        cudnnConvolutionDescriptor_t conv_desc;
        CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
        CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(
            conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        cudnnTensorDescriptor_t output_desc;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
            output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size, out_channels, output_height, output_width));

        // Use a safe fallback algorithm
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        size_t workspace_size = 0;
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_handle, input_desc, weight_desc, conv_desc, output_desc, algo, &workspace_size));
        void* workspace = nullptr;
        if (workspace_size > 0) {
            CHECK_CUDA_ERROR(cudaMalloc(&workspace, workspace_size));
        }

        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(
            cudnn_handle, &alpha, input_desc, input.data_ptr<float>(),
            weight_desc, weight.data_ptr<float>(), conv_desc, algo, workspace, workspace_size,
            &beta, output_desc, output.data_ptr<float>()));

        if (bias.defined()) {
            cudnnTensorDescriptor_t bias_desc;
            CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bias_desc));
            CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
                bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, out_channels, 1, 1));
            CHECK_CUDNN_ERROR(cudnnAddTensor(
                cudnn_handle, &alpha, bias_desc, bias.data_ptr<float>(),
                &alpha, output_desc, output.data_ptr<float>()));
            CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(bias_desc));
        }

        if (workspace) CHECK_CUDA_ERROR(cudaFree(workspace));
        CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
        CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
        CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(weight_desc));
        CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
        CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle));

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
        auto grad_bias = bias.defined() ? torch::zeros_like(bias) : torch::Tensor();

        cudnnHandle_t cudnn_handle;
        CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle));

        cudnnTensorDescriptor_t input_desc, output_desc;
        cudnnFilterDescriptor_t weight_desc;
        cudnnConvolutionDescriptor_t conv_desc;

        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
            input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size, in_channels, input_height, input_width));

        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
            output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            batch_size, out_channels, output_height, output_width));

        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&weight_desc));
        CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(
            weight_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
            out_channels, in_channels, kernel_height, kernel_width));

        CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));
        CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(
            conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

        cudnnConvolutionBwdDataAlgo_t bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        size_t bwd_data_workspace_size = 0;
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn_handle, weight_desc, output_desc, conv_desc, input_desc,
            bwd_data_algo, &bwd_data_workspace_size));
        void* bwd_data_workspace = nullptr;
        if (bwd_data_workspace_size > 0) {
            CHECK_CUDA_ERROR(cudaMalloc(&bwd_data_workspace, bwd_data_workspace_size));
        }

        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(
            cudnn_handle, &alpha, weight_desc, weight.data_ptr<float>(),
            output_desc, grad_output[0].data_ptr<float>(), conv_desc, bwd_data_algo,
            bwd_data_workspace, bwd_data_workspace_size, &beta, input_desc, grad_input.data_ptr<float>()));

        cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        size_t bwd_filter_workspace_size = 0;
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnn_handle, input_desc, output_desc, conv_desc, weight_desc,
            bwd_filter_algo, &bwd_filter_workspace_size));
        void* bwd_filter_workspace = nullptr;
        if (bwd_filter_workspace_size > 0) {
            CHECK_CUDA_ERROR(cudaMalloc(&bwd_filter_workspace, bwd_filter_workspace_size));
        }

        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(
            cudnn_handle, &alpha, input_desc, input.data_ptr<float>(),
            output_desc, grad_output[0].data_ptr<float>(), conv_desc, bwd_filter_algo,
            bwd_filter_workspace, bwd_filter_workspace_size, &beta, weight_desc, grad_weight.data_ptr<float>()));

        if (bias.defined()) {
            cudnnTensorDescriptor_t bias_desc;
            CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bias_desc));
            CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(
                bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                1, out_channels, 1, 1));
            CHECK_CUDNN_ERROR(cudnnConvolutionBackwardBias(
                cudnn_handle, &alpha, output_desc, grad_output[0].data_ptr<float>(),
                &beta, bias_desc, grad_bias.data_ptr<float>()));
            CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(bias_desc));
        }

        if (bwd_data_workspace) CHECK_CUDA_ERROR(cudaFree(bwd_data_workspace));
        if (bwd_filter_workspace) CHECK_CUDA_ERROR(cudaFree(bwd_filter_workspace));
        CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
        CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
        CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(weight_desc));
        CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
        CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle));

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor()};
    }
};

// Python 綁定
torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                           int stride, int padding) {
    return Conv2dCuDNNFunction::apply(input, weight, bias, stride, padding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d", &conv2d, "Custom 2D convolution using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("padding") = 0);
}