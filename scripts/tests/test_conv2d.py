import os

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import gradcheck

from cnn_methods import AVAILABLE_CNNs
from cnn_module.base import Conv2dBaseClass

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CONV2D_CLASSES: list[tuple[str, type[Conv2dBaseClass]]] = [
    (item[1]["short_name"], item[1]["class"]) for item in AVAILABLE_CNNs.items()
]

pytestmark = pytest.mark.parametrize("conv2d_name,Conv2d", CONV2D_CLASSES)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def input_tensor(device: torch.device) -> Tensor:
    return torch.randn(1, 7, 64, 64, device=device, requires_grad=True)


@pytest.mark.forward
def test_conv2d_correctness(input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass]) -> None:
    """
    測試自定義 Conv2d 是否與 PyTorch 官方實現的 Conv2d 在相同輸入下產生相同輸出。
    """
    torch.manual_seed(42)

    # Simplified input and parameters
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0
    input_tensor = torch.randn(1, in_channels, 5, 5, device=input_tensor.device)

    custom_conv = Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )

    reference_conv = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )

    if "cpu" not in conv2d_name:
        custom_conv.to(input_tensor.device)
        reference_conv.to(input_tensor.device)
    else:
        reference_conv.to("cpu")
        custom_conv.to("cpu")
        input_tensor = input_tensor.to("cpu")

    # Set known weights and bias
    with torch.no_grad():
        custom_conv.weight.fill_(1.0)
        custom_conv.bias.fill_(0.0)
        reference_conv.weight.copy_(custom_conv.weight)
        reference_conv.bias.copy_(custom_conv.bias)  # type: ignore

    custom_output = custom_conv(input_tensor)
    reference_output = reference_conv(input_tensor)

    # Debug: Print outputs
    print(f"Custom output:\n{custom_output}")
    print(f"Reference output:\n{reference_output}")

    assert torch.allclose(custom_output, reference_output, rtol=1e-5, atol=1e-8), (
        f"Outputs differ for {conv2d_name}: custom output shape {custom_output.shape}, "
        f"reference output shape {reference_output.shape}"
    )


# @pytest.mark.forward
# @pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
# def test_conv2d_parameters(input_tensor: Tensor, conv2d_name: str, Conv2d: type, stride: int, padding: int) -> None:
#     """
#     測試自定義 Conv2d 的輸出形狀是否符合預期。
#     """
#     in_channels = input_tensor.size(1)
#     conv = Conv2d(
#         in_channels=in_channels,
#         out_channels=16,
#         kernel_size=3,
#         stride=stride,
#         padding=padding,
#     )
#     if "cpu" not in conv2d_name:
#         conv.to(input_tensor.device)
#     else:
#         conv.to("cpu")
#         input_tensor = input_tensor.to("cpu")
#     output = conv(input_tensor)
#     expected_shape = (
#         input_tensor.size(0),
#         16,
#         (input_tensor.size(2) - 3 + 2 * padding) // stride + 1,
#         (input_tensor.size(3) - 3 + 2 * padding) // stride + 1,
#     )
#     assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape} for {conv2d_name}"


# @pytest.mark.error_handling
# def test_conv2d_cpu_error(conv2d_name: str, Conv2d: type[Conv2dBaseClass]) -> None:
#     """
#     測試自定義 Conv2d 在 CPU 上使用時是否會引發錯誤。
#     """
#     conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
#     input_tensor = torch.randn(1, 3, 64, 64)  # CPU tensor
#     with pytest.raises(Exception, match="Input must be a CUDA tensor"):
#         conv(input_tensor)


# @pytest.mark.backward
# def test_conv2d_backward_correctness(input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass]) -> None:
#     """
#     測試自定義 Conv2d 的 backward 傳播是否與 PyTorch 官方實現一致。
#     """
#     torch.manual_seed(42)

#     # 簡化輸入和參數以便測試
#     in_channels = 1
#     out_channels = 1
#     kernel_size = 3
#     stride = 1
#     padding = 0
#     input_tensor = torch.randn(1, in_channels, 5, 5, device=input_tensor.device, requires_grad=True, dtype=torch.float)

#     # 初始化自定義卷積和參考卷積
#     custom_conv = Conv2d(
#         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#     )

#     reference_conv = nn.Conv2d(
#         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#     )

#     if "cpu" not in conv2d_name:
#         custom_conv.to(input_tensor.device)
#         reference_conv.to(input_tensor.device)
#     else:
#         reference_conv.to("cpu")
#         custom_conv.to("cpu")
#         input_tensor = input_tensor.to("cpu")

#     # 設置相同的權重和偏置
#     with torch.no_grad():
#         custom_conv.weight.fill_(1.0)
#         custom_conv.bias.fill_(0.0)
#         reference_conv.weight.copy_(custom_conv.weight)
#         reference_conv.bias.copy_(custom_conv.bias)  # type: ignore

#     # Forward 傳播
#     custom_output = custom_conv(input_tensor)
#     reference_output = reference_conv(input_tensor)

#     # 檢查 Forward 結果是否一致
#     assert torch.allclose(custom_output, reference_output, rtol=1e-5, atol=1e-8), (
#         f"Forward outputs differ for {conv2d_name}: custom output shape {custom_output.shape}, "
#         f"reference output shape {reference_output.shape}"
#     )

#     # 設置隨機梯度作為輸出梯度
#     grad_output = torch.randn_like(custom_output)

#     # Backward 傳播
#     custom_output.backward(grad_output, retain_graph=True)
#     reference_output.backward(grad_output, retain_graph=True)

#     if (
#         custom_conv.weight.grad is None
#         or custom_conv.bias.grad is None
#         or input_tensor.grad is None
#         or reference_conv.weight.grad is None
#         or reference_conv.bias is None
#         or reference_conv.bias.grad is None
#     ):
#         raise AssertionError(f"Gradients are None for {conv2d_name}.")

#     # 檢查梯度
#     assert torch.allclose(input_tensor.grad, input_tensor.grad, rtol=1e-5, atol=1e-8), (
#         f"Input gradients differ for {conv2d_name}"
#     )
#     assert torch.allclose(custom_conv.weight.grad, reference_conv.weight.grad, rtol=1e-5, atol=1e-8), (
#         f"Weight gradients differ for {conv2d_name}"
#     )
#     assert torch.allclose(custom_conv.bias.grad, reference_conv.bias.grad, rtol=1e-5, atol=1e-8), (
#         f"Bias gradients differ for {conv2d_name}"
#     )


# @pytest.mark.backward
# @pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
# def test_conv2d_backward_shape(
#     input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass], stride: int, padding: int
# ) -> None:
#     """
#     測試自定義 Conv2d 的 backward 傳播梯度形狀是否正確。
#     """
#     in_channels = input_tensor.size(1)
#     out_channels = 16
#     kernel_size = 3

#     conv = Conv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#     )
#     if "cpu" not in conv2d_name:
#         conv.to(input_tensor.device)
#     else:
#         conv.to("cpu")
#         input_tensor = input_tensor.to("cpu")

#     # 確保輸入需要梯度
#     input_tensor.requires_grad_(True)

#     output = conv(input_tensor)
#     grad_output = torch.randn_like(output)
#     output.backward(grad_output)

#     # 檢查梯度形狀
#     expected_input_shape = input_tensor.shape
#     expected_weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
#     expected_bias_shape = (out_channels,)

#     if input_tensor.grad is None or conv.bias.grad is None or conv.weight.grad is None:
#         raise AssertionError(
#             f"Gradients are None for {conv2d_name}. Ensure requires_grad=True for input and parameters."
#         )

#     assert input_tensor.grad.shape == expected_input_shape, (
#         f"Input gradient shape {input_tensor.grad.shape} != expected {expected_input_shape} for {conv2d_name}"
#     )
#     assert conv.weight.grad.shape == expected_weight_shape, (
#         f"Weight gradient shape {conv.weight.grad.shape} != expected {expected_weight_shape} for {conv2d_name}"
#     )
#     assert conv.bias.grad.shape == expected_bias_shape, (
#         f"Bias gradient shape {conv.bias.grad.shape} != expected {expected_bias_shape} for {conv2d_name}"
#     )


# @pytest.mark.backward
# def test_conv2d_backward_gradcheck(conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device) -> None:
#     """
#     使用 torch.autograd.gradcheck 測試自定義 Conv2d 的梯度計算數值穩定性。
#     """
#     torch.manual_seed(42)

#     # 使用較小的輸入以加快 gradcheck，並使用 torch.float 匹配 CUDA 內核
#     in_channels = 1
#     out_channels = 1
#     kernel_size = 3
#     stride = 1
#     padding = 0
#     input_tensor = torch.randn(1, in_channels, 5, 5, device=device, dtype=torch.float, requires_grad=True)

#     conv = Conv2d(
#         in_channels=in_channels,
#         out_channels=out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         padding=padding,
#     )
#     conv.to(device).to(torch.float)

#     # 將參數設置為需要梯度
#     conv.weight.requires_grad_(True)
#     conv.bias.requires_grad_(True)

#     # 使用 gradcheck 測試梯度，調整容差以適應 float32
#     test = gradcheck(conv, (input_tensor,), eps=1e-4, atol=1e-3, rtol=1e-2)
#     assert test, f"Gradient check failed for {conv2d_name}"


# @pytest.mark.backward
# @pytest.mark.parametrize("stride,padding", [(2, 1), (3, 2), (1, 3)])
# def test_conv2d_backward_extended_parameters(
#     conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device, stride: int, padding: int
# ) -> None:
#     """
#     測試自定義 Conv2d 在不同 stride 和 padding 下的 backward 傳播正確性。
#     """
#     torch.manual_seed(42)

#     in_channels = 1
#     out_channels = 1
#     kernel_size = 3
#     input_tensor = torch.randn(1, in_channels, 7, 7, device=device, dtype=torch.float, requires_grad=True)

#     custom_conv = Conv2d(
#         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#     )

#     reference_conv = nn.Conv2d(
#         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#     )
#     if "cpu" not in conv2d_name:
#         custom_conv.to(device)
#         reference_conv.to(device)
#     else:
#         reference_conv.to("cpu")
#         custom_conv.to("cpu")
#         input_tensor = input_tensor.to("cpu")

#     if reference_conv.bias is None:
#         raise AssertionError(
#             f"Reference Conv2d {conv2d_name} does not have a bias term. Ensure bias=True in the constructor."
#         )

#     # 設置隨機權重和偏置
#     with torch.no_grad():
#         custom_conv.weight.copy_(torch.randn_like(custom_conv.weight))
#         custom_conv.bias.copy_(torch.randn_like(custom_conv.bias))
#         reference_conv.weight.copy_(custom_conv.weight)
#         reference_conv.bias.copy_(custom_conv.bias)

#     # Forward 傳播
#     custom_output = custom_conv(input_tensor)
#     reference_output = reference_conv(input_tensor)

#     # 檢查 Forward 結果
#     assert torch.allclose(custom_output, reference_output, rtol=1e-5, atol=1e-8), (
#         f"Forward outputs differ for {conv2d_name} with stride={stride}, padding={padding}"
#     )

#     # Backward 傳播
#     grad_output = torch.randn_like(custom_output)
#     custom_output.backward(grad_output, retain_graph=True)
#     reference_output.backward(grad_output, retain_graph=True)

#     if (
#         custom_conv.weight.grad is None
#         or custom_conv.bias.grad is None
#         or input_tensor.grad is None
#         or reference_conv.weight.grad is None
#         or reference_conv.bias.grad is None
#     ):
#         raise AssertionError(
#             f"Gradients are None for {conv2d_name}. Ensure requires_grad=True for input and parameters."
#         )

#     # 檢查梯度
#     assert torch.allclose(input_tensor.grad, input_tensor.grad, rtol=1e-5, atol=1e-8), (
#         f"Input gradients differ for {conv2d_name} with stride={stride}, padding={padding}"
#     )
#     assert torch.allclose(custom_conv.weight.grad, reference_conv.weight.grad, rtol=1e-5, atol=1e-8), (
#         f"Weight gradients differ for {conv2d_name} with stride={stride}, padding={padding}"
#     )
#     assert torch.allclose(custom_conv.bias.grad, reference_conv.bias.grad, rtol=1e-5, atol=1e-8), (
#         f"Bias gradients differ for {conv2d_name} with stride={stride}, padding={padding}"
#     )


# @pytest.mark.training
# def test_conv2d_training_step(conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device) -> None:
#     """
#     模擬單個訓練步驟，檢查梯度更新是否導致損失減少。
#     """
#     torch.manual_seed(42)

#     in_channels = 3
#     out_channels = 16
#     kernel_size = 3
#     stride = 1
#     padding = 1
#     input_tensor = torch.randn(4, in_channels, 32, 32, device=device, dtype=torch.float, requires_grad=True)
#     target = torch.randint(0, 10, (4,), device=device)

#     conv = Conv2d(
#         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#     )

#     if "cpu" not in conv2d_name:
#         conv.to(device)
#     else:
#         input_tensor = input_tensor.to("cpu")
#         conv.to("cpu")

#     # 簡單的線性層將卷積輸出轉換為分類
#     linear = nn.Linear(out_channels * 32 * 32, 10)
#     if "cpu" not in conv2d_name:
#         linear.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(list(conv.parameters()) + list(linear.parameters()), lr=0.01)

#     # 初始損失
#     conv_output = conv(input_tensor)
#     flat_output = conv_output.view(conv_output.size(0), -1)
#     logits = linear(flat_output)
#     initial_loss = criterion(logits, target)

#     # 訓練步驟
#     optimizer.zero_grad()
#     loss = criterion(logits, target)
#     loss.backward()
#     optimizer.step()

#     # 再次計算損失
#     conv_output = conv(input_tensor)
#     flat_output = conv_output.view(conv_output.size(0), -1)
#     logits = linear(flat_output)
#     new_loss = criterion(logits, target)

#     assert new_loss < initial_loss, (
#         f"Loss did not decrease after training step for {conv2d_name}: initial={initial_loss.item()}, new={new_loss.item()}"
#     )


# @pytest.mark.backward
# def test_conv2d_gradient_magnitude(conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device) -> None:
#     """
#     檢查梯度的量級是否合理，防止過大或過小的梯度。
#     """
#     torch.manual_seed(42)

#     in_channels = 3
#     out_channels = 16
#     kernel_size = 3
#     stride = 1
#     padding = 1
#     input_tensor = torch.randn(2, in_channels, 32, 32, device=device, dtype=torch.float, requires_grad=True)

#     conv = Conv2d(
#         in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#     )
#     if "cpu" not in conv2d_name:
#         conv.to(device)
#     else:
#         input_tensor = input_tensor.to("cpu")
#         conv.to("cpu")

#     output = conv(input_tensor)
#     grad_output = torch.randn_like(output)
#     output.backward(grad_output)

#     if input_tensor.grad is None or conv.weight.grad is None or conv.bias.grad is None:
#         raise AssertionError(
#             f"Gradients are None for {conv2d_name}. Ensure requires_grad=True for input and parameters."
#         )

#     # 檢查梯度量級
#     input_grad_norm = torch.norm(input_tensor.grad)
#     weight_grad_norm = torch.norm(conv.weight.grad)
#     bias_grad_norm = torch.norm(conv.bias.grad)

#     assert 1e-5 < input_grad_norm.item() < 1e5, (
#         f"Input gradient norm {input_grad_norm.item()} is out of reasonable range for {conv2d_name}"
#     )
#     assert 1e-5 < weight_grad_norm.item() < 1e5, (
#         f"Weight gradient norm {weight_grad_norm.item()} is out of reasonable range for {conv2d_name}"
#     )
#     assert 1e-5 < bias_grad_norm.item() < 1e5, (
#         f"Bias gradient norm {bias_grad_norm.item()} is out of reasonable range for {conv2d_name}"
#     )
