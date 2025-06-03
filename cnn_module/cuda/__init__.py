from .base import Conv2d as BaseConv2d

from .fft import Conv2d as FftConv2d
# from .img2col import Conv2d as Img2colConv2d

__all__ = ["BaseConv2d", "FftConv2d", "Img2colConv2d"]
