from __future__ import annotations
import torch
__all__ = ['conv2d', 'conv2d_share']
def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    Custom 2D Convolution (CUDA)
    """
def conv2d_share(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    Custom 2D Convolution (CUDA)
    """
