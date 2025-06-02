#!/bin/bash

# Set LD_LIBRARY_PATH to include PyTorch libraries for libc10.so
echo "Setting up PyTorch library path..."
TORCH_LIB_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')")
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH

# Set PYTHONPATH to include ./ for importing
echo "Setting PYTHONPATH to include ./..."
export PYTHONPATH=$(pwd):$PYTHONPATH