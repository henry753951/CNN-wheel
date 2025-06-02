from custom_cnn.cpu import _base
from torch import Tensor

from cnn_methods.base import Conv2dBaseClass


class Conv2d(Conv2dBaseClass):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        return _base.conv2d(x, self.weight, self.bias, self.stride, self.padding)
