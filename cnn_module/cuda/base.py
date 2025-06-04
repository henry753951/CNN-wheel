import torch.nn as nn
from torch import Tensor

from cnn_module.base import Conv2dBaseClass
from custom_cnn.cuda import _base


class Conv2d(Conv2dBaseClass):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_shared_memory: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.use_shared_memory = use_shared_memory
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor):
        if self.use_shared_memory:
            return _base.conv2d_share(x, self.weight, self.bias, self.stride, self.padding)
        return _base.conv2d(x, self.weight, self.bias, self.stride, self.padding)
