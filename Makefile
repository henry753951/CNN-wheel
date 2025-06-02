CUDA_ARCH := 7.5
export TORCH_CUDA_ARCH_LIST := $(CUDA_ARCH)
export PYTHONPATH=$PYTHONPATH:/workspaces/CNN-wheel
export CC := /usr/bin/gcc-11
export CXX := /usr/bin/g++-11

# 編譯 PyTorch 擴展
build:
	python3 setup.py build_ext --inplace

# 清理編譯文件
clean:
	rm -rf build *.so *.egg-info

# 運行卷積測試
test-conv:
	python3 -m unittest scripts/tests/test_conv2d.py

train-test:
	python3 scripts/train_test.py


# 運行所有測試
test: test-conv test-model

# 預設目標
all: build test