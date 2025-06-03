import pytest
import torch
import torch.nn as nn
from torch import Tensor

from cnn_module.base import Conv2dBaseClass

# List of Conv2d implementations to test
from cnn_module.cuda.base import Conv2d as Conv2dBase

# Filter out None values (for modules that may not exist)
CONV2D_CLASSES: list[tuple[str, type[Conv2dBaseClass]]] = [
    ("base", Conv2dBase),
    # ("fft", Conv2dFft),
]


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def input_tensor(device: torch.device) -> Tensor:
    return torch.randn(1, 7, 64, 64, device=device)


@pytest.mark.parametrize("conv2d_name,Conv2d", CONV2D_CLASSES)
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
    custom_conv.to(input_tensor.device)

    reference_conv = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    reference_conv.to(input_tensor.device)

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


@pytest.mark.parametrize("conv2d_name,Conv2d", CONV2D_CLASSES)
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
def test_conv2d_parameters(
    input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass], stride: int, padding: int
) -> None:
    """
    測試自定義 Conv2d 的輸出形狀是否符合預期。
    """
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=stride, padding=padding)
    conv.to(input_tensor.device)
    output = conv(input_tensor)
    expected_shape = (
        input_tensor.size(0),
        16,
        (input_tensor.size(2) - 3 + 2 * padding) // stride + 1,
        (input_tensor.size(3) - 3 + 2 * padding) // stride + 1,
    )
    assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape} for {conv2d_name}"


@pytest.mark.parametrize("conv2d_name,Conv2d", CONV2D_CLASSES)
def test_conv2d_cpu_error(conv2d_name: str, Conv2d: type[Conv2dBaseClass]) -> None:
    """
    測試自定義 Conv2d 在 CPU 上使用時是否會引發錯誤。
    """
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    input_tensor = torch.randn(1, 3, 64, 64)  # CPU tensor
    with pytest.raises(Exception, match="Input must be a CUDA tensor"):
        conv(input_tensor)
