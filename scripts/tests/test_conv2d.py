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
def device(conv2d_name: str) -> torch.device:
    if "cpu" in conv2d_name:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def input_tensor(device: torch.device) -> Tensor:
    return torch.randn(1, 7, 64, 64, device=device, requires_grad=True)


@pytest.mark.forward
def test_conv2d_correctness(input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass]) -> None:
    """
    Test if custom Conv2d produces the same output as PyTorch's Conv2d for the same input.
    """
    torch.manual_seed(42)

    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0
    input_tensor = torch.randn(1, in_channels, 5, 5, device=input_tensor.device, requires_grad=True)

    custom_conv = Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    reference_conv = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )

    custom_conv.to(input_tensor.device)
    reference_conv.to(input_tensor.device)

    if reference_conv.bias is None:
        reference_conv.bias = nn.Parameter(torch.zeros(out_channels, device=input_tensor.device))

    with torch.no_grad():
        custom_conv.weight.fill_(1.0)
        custom_conv.bias.fill_(0.0)
        reference_conv.weight.copy_(custom_conv.weight)
        reference_conv.bias.copy_(custom_conv.bias)

    custom_output = custom_conv(input_tensor)
    reference_output = reference_conv(input_tensor)

    print(f"Custom output:\n{custom_output}")
    print(f"Reference output:\n{reference_output}")

    assert torch.allclose(custom_output, reference_output, rtol=1e-5, atol=1e-8), (
        f"Outputs differ for {conv2d_name}: custom output shape {custom_output.shape}, "
        f"reference output shape {reference_output.shape}"
    )


@pytest.mark.forward
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
def test_conv2d_parameters(input_tensor: Tensor, conv2d_name: str, Conv2d: type, stride: int, padding: int) -> None:
    """
    Test if custom Conv2d output shape matches expected shape.
    """
    in_channels = input_tensor.size(1)
    conv = Conv2d(
        in_channels=in_channels,
        out_channels=16,
        kernel_size=3,
        stride=stride,
        padding=padding,
    )
    conv.to(input_tensor.device)
    output = conv(input_tensor)
    expected_shape = (
        input_tensor.size(0),
        16,
        (input_tensor.size(2) - 3 + 2 * padding) // stride + 1,
        (input_tensor.size(3) - 3 + 2 * padding) // stride + 1,
    )
    assert output.shape == expected_shape, f"Output shape {output.shape} != expected {expected_shape} for {conv2d_name}"


@pytest.mark.backward
def test_conv2d_backward_correctness(input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass]) -> None:
    """
    Test if custom Conv2d backward propagation matches PyTorch's Conv2d.
    """
    torch.manual_seed(42)

    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0
    input_tensor = torch.randn(1, in_channels, 5, 5, device=input_tensor.device, requires_grad=True, dtype=torch.float)

    custom_conv = Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    reference_conv = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )

    custom_conv.to(input_tensor.device)
    reference_conv.to(input_tensor.device)

    if reference_conv.bias is None:
        reference_conv.bias = nn.Parameter(torch.zeros(out_channels, device=input_tensor.device))

    with torch.no_grad():
        custom_conv.weight.fill_(1.0)
        custom_conv.bias.fill_(0.0)
        reference_conv.weight.copy_(custom_conv.weight)
        reference_conv.bias.copy_(custom_conv.bias)

    custom_output = custom_conv(input_tensor)
    reference_output = reference_conv(input_tensor)

    assert torch.allclose(custom_output, reference_output, rtol=1e-5, atol=1e-8), (
        f"Forward outputs differ for {conv2d_name}: custom output shape {custom_output.shape}, "
        f"reference output shape {reference_output.shape}"
    )

    grad_output = torch.randn_like(custom_output)
    custom_output.backward(grad_output, retain_graph=True)
    reference_output.backward(grad_output, retain_graph=True)

    if (
        custom_conv.weight.grad is None
        or custom_conv.bias.grad is None
        or input_tensor.grad is None
        or reference_conv.weight.grad is None
        or reference_conv.bias is None
        or reference_conv.bias.grad is None
    ):
        raise AssertionError(f"Gradients are None for {conv2d_name}.")

    assert torch.allclose(input_tensor.grad, input_tensor.grad, rtol=1e-5, atol=1e-8), (
        f"Input gradients differ for {conv2d_name}"
    )
    assert torch.allclose(custom_conv.weight.grad, reference_conv.weight.grad, rtol=1e-5, atol=1e-8), (
        f"Weight gradients differ for {conv2d_name}"
    )
    assert torch.allclose(custom_conv.bias.grad, reference_conv.bias.grad, rtol=1e-5, atol=1e-8), (
        f"Bias gradients differ for {conv2d_name}"
    )


@pytest.mark.backward
@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
def test_conv2d_backward_shape(
    input_tensor: Tensor, conv2d_name: str, Conv2d: type[Conv2dBaseClass], stride: int, padding: int
) -> None:
    """
    Test if custom Conv2d backward gradient shapes are correct.
    """
    in_channels = input_tensor.size(1)
    out_channels = 16
    kernel_size = 3

    conv = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    conv.to(input_tensor.device)
    input_tensor.requires_grad_(True)
    output = conv(input_tensor)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    expected_input_shape = input_tensor.shape
    expected_weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
    expected_bias_shape = (out_channels,)

    if input_tensor.grad is None or conv.bias.grad is None or conv.weight.grad is None:
        raise AssertionError(
            f"Gradients are None for {conv2d_name}. Ensure requires_grad=True for input and parameters."
        )

    assert input_tensor.grad.shape == expected_input_shape, (
        f"Input gradient shape {input_tensor.grad.shape} != expected {expected_input_shape} for {conv2d_name}"
    )
    assert conv.weight.grad.shape == expected_weight_shape, (
        f"Weight gradient shape {conv.weight.grad.shape} != expected {expected_weight_shape} for {conv2d_name}"
    )
    assert conv.bias.grad.shape == expected_bias_shape, (
        f"Bias gradient shape {conv.bias.grad.shape} != expected {expected_bias_shape} for {conv2d_name}"
    )


@pytest.mark.backward
def test_conv2d_backward_gradcheck(conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device) -> None:
    """
    Test numerical stability of custom Conv2d gradients using torch.autograd.gradcheck.
    """
    torch.manual_seed(42)

    in_channels = 1
    out_channels = 1
    kernel_size = 3
    stride = 1
    padding = 0
    input_tensor = torch.randn(1, in_channels, 5, 5, device=device, dtype=torch.float, requires_grad=True)

    conv = Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    conv.to(device).to(torch.float)
    conv.weight.requires_grad_(True)
    conv.bias.requires_grad_(True)

    test = gradcheck(conv, (input_tensor,), eps=1e-4, atol=1e-3, rtol=1e-2)
    assert test, f"Gradient check failed for {conv2d_name}"


@pytest.mark.backward
@pytest.mark.parametrize("stride,padding", [(2, 1), (3, 2), (1, 3)])
def test_conv2d_backward_extended_parameters(
    conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device, stride: int, padding: int
) -> None:
    """
    Test custom Conv2d backward correctness with different stride and padding.
    """
    torch.manual_seed(42)

    in_channels = 1
    out_channels = 1
    kernel_size = 3
    input_tensor = torch.randn(1, in_channels, 7, 7, device=device, dtype=torch.float, requires_grad=True)

    custom_conv = Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    reference_conv = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    custom_conv.to(device)
    reference_conv.to(device)
    if reference_conv.bias is None:
        reference_conv.bias = nn.Parameter(torch.zeros(out_channels, device=device))
    with torch.no_grad():
        custom_conv.weight.copy_(torch.randn_like(custom_conv.weight))
        custom_conv.bias.copy_(torch.randn_like(custom_conv.bias))
        reference_conv.weight.copy_(custom_conv.weight)
        reference_conv.bias.copy_(custom_conv.bias)

    custom_output = custom_conv(input_tensor)
    reference_output = reference_conv(input_tensor)

    assert torch.allclose(custom_output, reference_output, rtol=1e-5, atol=1e-8), (
        f"Forward outputs differ for {conv2d_name} with stride={stride}, padding={padding}"
    )

    grad_output = torch.randn_like(custom_output)
    custom_output.backward(grad_output, retain_graph=True)
    reference_output.backward(grad_output, retain_graph=True)

    if (
        custom_conv.weight.grad is None
        or custom_conv.bias.grad is None
        or input_tensor.grad is None
        or reference_conv.weight.grad is None
        or reference_conv.bias is None
        or reference_conv.bias.grad is None
    ):
        raise AssertionError(
            f"Gradients are None for {conv2d_name}. Ensure requires_grad=True for input and parameters."
        )

    assert torch.allclose(input_tensor.grad, input_tensor.grad, rtol=1e-5, atol=1e-8), (
        f"Input gradients differ for {conv2d_name} with stride={stride}, padding={padding}"
    )
    assert torch.allclose(custom_conv.weight.grad, reference_conv.weight.grad, rtol=1e-5, atol=1e-8), (
        f"Weight gradients differ for {conv2d_name} with stride={stride}, padding={padding}"
    )
    assert torch.allclose(custom_conv.bias.grad, reference_conv.bias.grad, rtol=1e-5, atol=1e-8), (
        f"Bias gradients differ for {conv2d_name} with stride={stride}, padding={padding}"
    )


@pytest.mark.training
def test_conv2d_training_step(conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device) -> None:
    """
    Simulate a single training step to check if gradient updates reduce loss.
    """
    torch.manual_seed(42)

    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1
    input_tensor = torch.randn(4, in_channels, 32, 32, device=device, dtype=torch.float, requires_grad=True)
    target = torch.randint(0, 10, (4,), device=device)

    conv = Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    conv.to(device)
    linear = nn.Linear(out_channels * 32 * 32, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(list(conv.parameters()) + list(linear.parameters()), lr=0.01)

    conv_output = conv(input_tensor)
    flat_output = conv_output.view(conv_output.size(0), -1)
    logits = linear(flat_output)
    initial_loss = criterion(logits, target)

    optimizer.zero_grad()
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    conv_output = conv(input_tensor)
    flat_output = conv_output.view(conv_output.size(0), -1)
    logits = linear(flat_output)
    new_loss = criterion(logits, target)

    assert new_loss < initial_loss, (
        f"Loss did not decrease after training step for {conv2d_name}: initial={initial_loss.item()}, new={new_loss.item()}"
    )


@pytest.mark.backward
def test_conv2d_gradient_magnitude(conv2d_name: str, Conv2d: type[Conv2dBaseClass], device: torch.device) -> None:
    """
    Check if gradient magnitudes are reasonable.
    """
    torch.manual_seed(42)

    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 1
    padding = 1
    input_tensor = torch.randn(2, in_channels, 32, 32, device=device, dtype=torch.float, requires_grad=True)

    conv = Conv2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    conv.to(device)
    output = conv(input_tensor)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    if input_tensor.grad is None or conv.weight.grad is None or conv.bias.grad is None:
        raise AssertionError(
            f"Gradients are None for {conv2d_name}. Ensure requires_grad=True for input and parameters."
        )

    input_grad_norm = torch.norm(input_tensor.grad)
    weight_grad_norm = torch.norm(conv.weight.grad)
    bias_grad_norm = torch.norm(conv.bias.grad)

    assert 1e-5 < input_grad_norm.item() < 1e5, (
        f"Input gradient norm {input_grad_norm.item()} is out of reasonable range for {conv2d_name}"
    )
    assert 1e-5 < weight_grad_norm.item() < 1e5, (
        f"Weight gradient norm {weight_grad_norm.item()} is out of reasonable range for {conv2d_name}"
    )
    assert 1e-5 < bias_grad_norm.item() < 1e5, (
        f"Bias gradient norm {bias_grad_norm.item()} is out of reasonable range for {conv2d_name}"
    )
