from __future__ import annotations
import torch
__all__ = ['conv2d']
def conv2d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """
    A custom CPU-based Conv2d operation (for educational purposes)
    """
