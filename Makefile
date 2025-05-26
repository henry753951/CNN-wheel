# 編譯 PyTorch 擴展
build:
	python setup.py build_ext --inplace

# 清理編譯文件
clean:
	rm -rf build *.so *.egg-info

# 運行卷積測試
test-conv:
	python -m unittest scripts/tests/test_conv2d.py

# 運行所有測試
test: test-conv test-model

# 預設目標
all: build test