CUDA_ARCH := 7.5
export TORCH_CUDA_ARCH_LIST := $(CUDA_ARCH)
export PYTHONPATH=$(pwd)/dist:$PYTHONPATH
export CC := /usr/bin/gcc-11
export CXX := /usr/bin/g++-11


build: 
	@echo "Building PyTorch extension..."
	chmod +x ./scripts/build.sh
	./scripts/build.sh

clean:
	rm -rf build *.so *.egg-info


test-conv:
	@echo "Running unit tests for conv2d..."
	chmod +x ./scripts/test_conv.sh
	./scripts/test_conv.sh

test:
	test-conv

all: build test