# CNN-wheel
```
export PYTHONPATH=$(pwd)/dist:$PYTHONPATH
```

## `cnn_methods` 封裝 CNN Class 的地方
定義了多種 CNN Class, 綁定到 custom_cnn 這個 package 中。 讓後續 Model 定義可以直接使用。

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


# Getting Started
## 1. 撰寫 cuda or cpp
位於 `src/` 內，撰寫完畢後

## 2. 設定 `setup.py`
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

## 3. 編譯
```bash
make build
```

## 4. 安裝
```bash
make install
```
# CNN-wheel