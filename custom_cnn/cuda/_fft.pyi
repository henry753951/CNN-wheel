from __future__ import annotations
import torch
__all__ = ['conv2d']
def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 0, use_fft: bool = False) -> torch.Tensor:
    """
    Custom 2D convolution (CUDA, FFT support)
    """
