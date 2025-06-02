import torch

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device available")
print(torch.cuda.current_device() if torch.cuda.is_available() else "No current CUDA device")
print(torch.cuda.device_count() if torch.cuda.is_available() else "No CUDA devices available")
print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "No CUDA device capability available")
print(torch.cuda.get_device_properties(0) if torch.cuda.is_available() else "No CUDA device properties available")
