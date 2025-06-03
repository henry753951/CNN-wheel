# CNN-wheel
```
# 手動將 module 引入到 Python 環境中
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## `src` 放置 C++ 或 CUDA 的原始碼
```
src/
├── cuda
│   ├── base
│   │   └── conv2d_base.cu
│   └── fft
│       └── conv2d_fft.cu
└── cpp
    └── base
        └── conv2d_base.cpp
```
## `cnn_module` 放置 Python 的 Module
```
cnn_module/
├── cuda
│   ├── base
│   │   └── conv2d.py
│   └── fft
│       └── conv2d.py
└── cpp
    └── base
        └── conv2d.py
``` 

## `models` 定義 Model 的地方
```
class CustomCNN(nn.Module):
    def __init__(self, cnn_type: str, num_classes: int = 10):
        super(CustomCNN, self).__init__()
        self.cnn = custom_cnn.get_cnn(cnn_type, num_classes)

    def forward(self, x):
        return self.cnn(x)
```

## `models/bin` 放置 Model 的 checkpoint
```
xxxx.pth
```

# Development Guide
## CUDA or C++ Extension Development
### 1. 撰寫 cuda or cpp
位於 `src/` 內，撰寫 CUDA 或 C++ 的原始碼檔案，例如 `src/cuda/base/conv2d_base.cu` 或 `src/cpp/base/conv2d_base.cpp`。

### 2. 設定 `setup.py`
```python
# setup.py
setup(
    ...
    ext_modules=[
        # 將需要編譯的 CUDA 或 C++ 檔案加入
        CUDAExtension(
            name="custom_cnn.cuda._base",  # 對應 custom_cnn.cuda._base
            sources=["src/cuda/base/conv2d_base.cu"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
    ]
    ...
)
```

### 3. 編譯
```bash
make build
```
## 包裝上面動態連結的 Function 成為 nn.Module class
### 1. 撰寫 Module
在 `cnn_module/cuda/base/conv2d.py` 中撰寫新的 Module，繼承自 `nn.Module`。
```python
import torch
from torch import nn
from custom_cnn.cuda._base import conv2d_base  # 對應 src/cuda/base/conv2d_base.cu 動態連接的函數
from cnn_module.cuda.base import Conv2dBaseClass
class Conv2dBase(Conv2dBaseClass):
    # 現有 init 參數名不可變更，僅可在後方添加其他參數
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dBase, self).__init__(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return conv2d_base(x, self.weight, self.bias, self.stride, self.padding)
```
### 2. 設定 `__init__.py`

```python
# cnn_module/cuda/__init__.py
from .base import Conv2d as BaseConv2d
from .fft import Conv2d as FftConv2d
from .img2col import Conv2d as Img2colConv2d

__all__ = ["BaseConv2d", "FftConv2d", "Img2colConv2d"]

```

## Testing Conv2d
### 1. 撰寫測試
在 `scripts/tests/test_conv2d.py` 中引入新的 CUDA 或 C++ 函數的測試。
```python
...

# List of Conv2d implementations to test
from cnn_module.cuda.base import Conv2d as Conv2dBase
from cnn_module.cuda.fft import Conv2d as Conv2dFft # 像是這裡

# Filter out None values (for modules that may not exist)
CONV2D_CLASSES: list[tuple[str, type[Conv2dBaseClass]]] = [
    ("base", Conv2dBase),
    ("fft", Conv2dFft), # 像是這裡
]

...
```

### 2. 執行測試
```bash
make test-conv
```


## Training
### 1. 編輯 `scripts/train/train.py` 中可用的 CNN 模組
```python
# scripts/train/train.py
...
from cnn_module.cuda.base import Conv2d as Conv2dBase
# from cnn_module.cuda.fft import Conv2d as Conv2dFft
# from cnn_module.cuda.img2col import Conv2d as Conv2dImg2Col
# 取消註解或添加其他 CNN 模組

AVAILABLE_CNNs = {
    "Official PyTorch": {"class": nn.Conv2d, "short_name": "official"},
    "Cuda Base": {"class": Conv2dBase, "short_name": "base"},
    # "Cuda FFT": {"class": Conv2dFft, "short_name": "fft"},
    # "Cuda Img2Col": {"class": Conv2dImg2Col, "short_name": "img2col"},
    # 取消註解或添加其他 CNN 模組
}

HYPER_PARAMETERS = {"learning_rate": 0.001, "batch_size": 128, "epochs": 10, "seed": 42, "val_split": 0.2}
...

```

### 2. 訓練模型
```bash
python3 scripts/train/train.py
```

## Inference
### 1. 編輯 `scripts/inference/inference.py` 中可用的 CNN 模組
```python
# scripts/inference/inference.py
...
from cnn_module.cuda.base import Conv2d as Conv2dBase
# from cnn_module.cuda.fft import Conv2d as Conv2dFft
# from cnn_module.cuda.img2col import Conv2d as Conv2dImg2Col

AVAILABLE_CNNs = {
    "Official PyTorch": {"class": nn.Conv2d, "short_name": "official"},
    "Cuda Base": {"class": Conv2dBase, "short_name": "base"},
    # "Cuda FFT": {"class": Conv2dFft, "short_name": "fft"},
    # "Cuda Img2Col": {"class": Conv2dImg2Col, "short_name": "img2col"},
}

```


## Model usage (我已經定義一個可以傳入不同 CNN 的 Model了，所以應該算暫時不用管吧)
### 1. 引入 Module
```python
from cnn_module.cuda.base import Conv2d as Conv2dBase
from cnn_module.cuda.fft import Conv2d as Conv2dFft
```

### 2. 在 Model 中使用 module
```python
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = Conv2dBase(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2dFft(16, 32, kernel_size=3, stride=1, padding=1)
        # 其他層...

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 其他操作...
        return x
```
