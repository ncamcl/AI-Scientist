import torch
   # 检查CUDA是否可用，如果不可用则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")
    
device = torch.device("mps")
# model = model.to(device)
# tensor = tensor.to(device)