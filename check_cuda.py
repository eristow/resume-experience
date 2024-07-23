import torch

print("PyTorch Version: ", torch.__version__)
print("CUDA Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
    print("CUDA Device Count: ", torch.cuda.device_count())
else:
    print("CUDA is not available. Check your CUDA installation and setup.")

