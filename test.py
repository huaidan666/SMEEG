import torch
print(torch.backends.cudnn.version())
if torch.cuda.is_available():
    # 查看 PyTorch 编译时使用的 CUDA 版本
    print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
    # 查看当前 GPU 的名称
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
