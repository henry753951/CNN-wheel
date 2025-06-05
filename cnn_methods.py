from torch import nn

from cnn_module.cpu.base import Conv2d as Conv2dBaseCPU
from cnn_module.cuda.base import Conv2d as Conv2dBaseCUDA
from cnn_module.cuda.fft import Conv2d as Conv2dFft
from cnn_module.cuda.img2col import Conv2d as Conv2dImg2col

AVAILABLE_CNNs = {
    "Official PyTorch": {"class": nn.Conv2d, "short_name": "official"},
    "Cuda Base": {"class": Conv2dBaseCUDA, "short_name": "cuda_base", "args": {"use_shared_memory": False}},
    "CPU Base": {"class": Conv2dBaseCPU, "short_name": "cpu_base"},
    # "Cuda FFT": {"class": Conv2dFft, "short_name": "cuda_fft"},
    # "Cuda Img2Col": {"class": Conv2dImg2col, "short_name": "cuda_img2col"},
}
