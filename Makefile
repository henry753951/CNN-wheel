CUDA_ARCH := 7.5
export TORCH_CUDA_ARCH_LIST := $(CUDA_ARCH)
export PYTHONPATH=$(pwd)/dist:$PYTHONPATH
export CC := /usr/bin/gcc-11
export CXX := /usr/bin/g++-11


build: 
	@echo "Building PyTorch extension..."
	make uninstall
	make install

clean:
	rm -rf build *.so *.egg-info

install:
	@echo "Installing PyTorch extension..."
	python3 setup.py install

uninstall:
	@echo "Uninstalling PyTorch extension..."
	pip uninstall -y custom-cnn

test-conv:
	@echo "Running unit tests for conv2d..."
	chmod +x ./scripts/test_conv.sh
	./scripts/test_conv.sh

test:
	test-conv

all: build test