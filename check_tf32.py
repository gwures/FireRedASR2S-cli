# check_tf32.py
import torch

# 1. 查看当前状态
print("原始状态 - CUDA matmul TF32开启状态:", torch.backends.cuda.matmul.allow_tf32)
print("原始状态 - CuDNN TF32开启状态:", torch.backends.cudnn.allow_tf32)

# 2. 统一开启 TF32（推荐，提升GPU计算速度）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# （如果想关闭，就设为 False：）
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# 3. 查看修改后的状态
print("\n修改后状态 - CUDA matmul TF32开启状态:", torch.backends.cuda.matmul.allow_tf32)
print("修改后状态 - CuDNN TF32开启状态:", torch.backends.cudnn.allow_tf32)