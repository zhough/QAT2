import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        #self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        self.in_features = 16 * 4 * 4
        self.fc1 = nn.Linear(self.in_features, 10)

    def relu_k(self, x, k=0.5):
        return torch.clamp(x, min=0, max=k)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_k(x)
        x = self.pool(x)
        #x = self.conv2(x)
        #x = self.relu_k(x)
        x = x.reshape(-1, self.in_features)
        x = self.fc1(x)
        return x
    
    def save_int8_weights(self, file_path):
        # 以INT8格式保存权重和偏置
        state_dict = self.state_dict()
        save_dict = {}
        
        for key, tensor in state_dict.items():
            # 将权重和偏置转换为INT8格式
            int8_tensor, scale = self.to_int8(tensor)
            # 分别保存数据和缩放因子
            save_dict[f"{key}_data"] = int8_tensor.cpu().numpy()
            save_dict[f"{key}_scale"] = scale
        
        # 保存为.npz文件
        np.savez(file_path, **save_dict)
        print(f"int8权重已保存至 {file_path}")
    
    def to_int8(self,tensor):
        # PyTorch中没有直接的FP8支持，这里使用自定义的FP8模拟
        # 使用INT8模拟FP8，但保持其动态范围
        # 确保我们使用的是最大绝对值得比例因子
        max_val = torch.max(torch.abs(tensor)).item()
        scale = 127.0 / max(max_val, 1e-10)  # 防止除以零
        
        # 量化到int8范围
        int8_tensor = torch.round(tensor * scale).clamp(-128, 127).to(torch.int8)
        return int8_tensor, scale


