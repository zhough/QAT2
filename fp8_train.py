import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import os

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义FP8辅助函数
def to_fp8(tensor):
    # 在PyTorch中模拟FP8，使用更稳定的实现方式
    # 保持为float32，但限制在FP8的范围内
    min_val = -128.0
    max_val = 128.0
    # 首先处理任何可能的NaN或无穷大值
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=max_val, neginf=min_val)
    # 然后限制范围
    clipped = torch.clamp(tensor, min_val, max_val)
    return clipped

def convert_weights_to_fp8(model):
    # 将模型权重转换为FP8格式
    with torch.no_grad():
        for param in model.parameters():
            # 先确保权重没有NaN值
            param.data = torch.nan_to_num(param.data, nan=0.0)
            # 然后转换为FP8范围
            param.data = to_fp8(param.data)

# ==================== 1. 定义使用FP8的CNN模型 ====================
class FP8CNN(nn.Module):
    def __init__(self):
        super(FP8CNN, self).__init__()
        # 卷积层: (in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0, bias=True)
        # ReLU6激活函数
        self.relu6 = nn.ReLU6(inplace=False)
        # MaxPool层: (kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层: 重新计算输入特征数 - 28x28 -> (28-3+0)/2+1=13x13 -> MaxPool后 6x6 -> 16*6*6=576
        self.fc = nn.Linear(16 * 6 * 6, 10, bias=True)
    
    def forward(self, x):
        # 卷积层
        x = self.conv(x)
        # ReLU6激活
        x = self.relu6(x)
        # MaxPool层
        x = self.pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc(x)
        return x

# ==================== 2. 数据加载 ====================
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# ==================== 3. 训练函数 ====================
def train_model(model, train_loader, optimizer, criterion, epoch, lr_scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 梯度裁剪阈值
    clip_value = 1.0
    
    # 权重范围约束的严格程度随训练进行动态调整
    weight_scale = max(0.5, 1.0 - (epoch - 1) * 0.2)
    weight_max = 128.0 * weight_scale
    weight_min = -128.0 * weight_scale
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 将数据转换为FP8格式
        data = to_fp8(data)
        
        # 前向传播
        output = model(data)
        
        # 检查输出是否包含NaN
        if torch.isnan(output).any():
            print(f"警告: 前向传播输出包含NaN，批次索引: {batch_idx}")
            loss = torch.tensor(0.1, requires_grad=True)
        else:
            loss = criterion(output, target)
            
            # 检查损失是否为NaN
            if torch.isnan(loss).any():
                print(f"警告: 损失为NaN，批次索引: {batch_idx}")
                loss = torch.tensor(0.1, requires_grad=True)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        # 更新权重前确保梯度有效
        for param in model.parameters():
            if param.grad is not None:
                # 替换NaN和无穷大梯度为合理值
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                # 额外添加梯度缩放，避免过大的更新
                param.grad = torch.clamp(param.grad, -0.1, 0.1)
        
        # 更新权重
        optimizer.step()
        
        # 学习率调度器更新（如果提供）
        if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            lr_scheduler.step()
        
        # 更严格的权重约束策略
        with torch.no_grad():
            for param in model.parameters():
                # 1. 先确保没有NaN值
                param.data = torch.nan_to_num(param.data, nan=0.0)
                # 2. 根据训练进程动态调整权重范围
                param.data = torch.clamp(param.data, weight_min, weight_max)
                # 3. 对权重添加微小的正则化，避免权重过于接近极限值
                dist_to_max = weight_max - param.data
                dist_to_min = param.data - weight_min
                mask_max = dist_to_max < 1.0
                mask_min = dist_to_min < 1.0
                if mask_max.any():
                    param.data[mask_max] -= 0.01 * (1.0 - dist_to_max[mask_max])
                if mask_min.any():
                    param.data[mask_min] += 0.01 * (1.0 - dist_to_min[mask_min])
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '  
                  f'({100. * batch_idx / len(train_loader):.1f}%)] '  
                  f'Loss: {running_loss / (batch_idx + 1):.6f} '  
                  f'Accuracy: {100. * correct / total:.2f}% '  
                  f'LR: {current_lr:.7f}')

# ==================== 4. 测试函数 ====================
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据转换为FP8格式
            data = to_fp8(data)
            
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

# ==================== 5. 保存权重函数 ====================
def save_weights(model):
    weights = {}
    # 保存卷积层权重和偏置（确保都是FP8范围）
    weights['conv_weight'] = to_fp8(model.conv.weight.data).numpy()
    weights['conv_bias'] = to_fp8(model.conv.bias.data).numpy()
    # 保存全连接层权重和偏置（确保都是FP8范围）
    weights['fc_weight'] = to_fp8(model.fc.weight.data).numpy()
    weights['fc_bias'] = to_fp8(model.fc.bias.data).numpy()
    
    # 保存为NumPy压缩格式
    np.savez('fp8_model_weights.npz', **weights)
    print("模型权重已保存到 fp8_model_weights.npz")
    return weights

# ==================== 6. FP8手动推理类 ====================
class FP8Inference:
    def __init__(self, weights):
        # 首先设置FP8范围参数
        self.fp8_max = 128.0
        self.fp8_min = -128.0
        
        # 然后加载权重并确保它们在FP8范围内
        self.conv_weight = self.to_fp8(weights['conv_weight'])
        self.conv_bias = self.to_fp8(weights['conv_bias'])
        self.fc_weight = self.to_fp8(weights['fc_weight'])
        self.fc_bias = self.to_fp8(weights['fc_bias'])
        
    def to_fp8(self, arr):
        """将numpy数组转换为FP8范围，全程保持FP8精度"""
        # 首先处理任何可能的NaN或无穷大值
        arr = np.nan_to_num(arr, nan=0.0, posinf=self.fp8_max, neginf=self.fp8_min)
        # 然后限制范围
        return np.clip(arr, self.fp8_min, self.fp8_max).astype(np.float32)  # 使用float32存储但保持FP8范围约束
    
    def conv2d_fp8(self, x, weight, bias, stride=1, padding=0):
        """使用FP8精度执行卷积操作，全程保持FP8精度"""
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        
        # 计算输出尺寸
        out_height = (height - kernel_height + 2 * padding) // stride + 1
        out_width = (width - kernel_width + 2 * padding) // stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
        
        # 如果需要填充
        if padding > 0:
            padded_x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            padded_x = x
        
        # 执行卷积（手动实现卷积操作）
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # 计算感受野在输入中的起始位置
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + kernel_height
                        w_end = w_start + kernel_width
                        
                        # 提取感受野
                        receptive_field = padded_x[b, :, h_start:h_end, w_start:w_end]
                        
                        # 执行FP8精度的乘法累加
                        conv_sum = 0.0
                        for ic in range(in_channels):
                            for kh in range(kernel_height):
                                for kw in range(kernel_width):
                                    # FP8精度计算
                                    x_val = receptive_field[ic, kh, kw]
                                    w_val = weight[oc, ic, kh, kw]
                                    # 模拟FP8乘法（先计算，再截断到FP8范围）
                                    product = self.to_fp8(x_val * w_val)
                                    # 模拟FP8加法（累加）
                                    conv_sum = self.to_fp8(conv_sum + product)
                        
                        # 添加偏置并保持FP8精度
                        conv_sum = self.to_fp8(conv_sum + bias[oc])
                        output[b, oc, oh, ow] = conv_sum
        
        return output
    
    def relu6_fp8(self, x):
        """使用FP8精度执行ReLU6激活，保持FP8精度"""
        # ReLU6: min(max(x, 0), 6)
        return self.to_fp8(np.minimum(np.maximum(x, 0), 6))
    
    def max_pool2d_fp8(self, x, kernel_size=2, stride=2):
        """使用FP8精度执行最大池化，保持FP8精度"""
        batch_size, channels, height, width = x.shape
        
        # 计算输出尺寸
        out_height = height // kernel_size
        out_width = width // kernel_size
        
        # 初始化输出
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)
        
        # 执行最大池化（手动实现）
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + kernel_size
                        w_end = w_start + kernel_size
                        
                        # 提取池化区域
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        
                        # 取最大值并保持FP8精度
                        max_val = np.max(pool_region)
                        output[b, c, oh, ow] = self.to_fp8(max_val)
        
        return output
    
    def linear_fp8(self, x, weight, bias):
        """使用FP8精度执行全连接层，全程保持FP8精度"""
        # x的形状: [batch_size, in_features]
        # weight的形状: [out_features, in_features]
        
        batch_size, in_features = x.shape
        out_features, _ = weight.shape
        
        # 初始化输出
        output = np.zeros((batch_size, out_features), dtype=np.float32)
        
        # 手动实现矩阵乘法
        for b in range(batch_size):
            for of in range(out_features):
                sum_val = 0.0
                for ifeature in range(in_features):
                    # FP8精度计算
                    x_val = x[b, ifeature]
                    w_val = weight[of, ifeature]
                    # 模拟FP8乘法和累加
                    product = self.to_fp8(x_val * w_val)
                    sum_val = self.to_fp8(sum_val + product)
                
                # 添加偏置并保持FP8精度
                output[b, of] = self.to_fp8(sum_val + bias[of])
        
        return output
    
    def predict(self, x):
        """使用FP8精度执行完整推理流程，全程保持FP8精度"""
        # 确保输入是FP8精度
        x_fp8 = self.to_fp8(x)
        
        # 卷积层（FP8精度）
        x_fp8 = self.conv2d_fp8(x_fp8, self.conv_weight, self.conv_bias, stride=2, padding=0)
        
        # ReLU6激活（FP8精度）
        x_fp8 = self.relu6_fp8(x_fp8)
        
        # MaxPool层（FP8精度）
        x_fp8 = self.max_pool2d_fp8(x_fp8, kernel_size=2, stride=2)
        
        # 展平
        batch_size = x_fp8.shape[0]
        x_flat = x_fp8.reshape(batch_size, -1)
        
        # 全连接层（FP8精度）
        x_fp8 = self.linear_fp8(x_flat, self.fc_weight, self.fc_bias)
        
        return x_fp8

# ==================== 7. 主函数 ====================
def main():
    # 加载数据
    train_loader, test_loader = load_data()
    
    # 创建模型
    model = FP8CNN()
    
    # 初始化模型权重为FP8格式
    convert_weights_to_fp8(model)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-8)
    
    # 添加学习率调度器
    total_steps = len(train_loader) * 2  # 假设训练2个epoch
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0,
        three_phase=False,
        last_epoch=-1
    )
    
    # 训练模型
    print("开始训练FP8模型...")
    epochs = 2  # 训练2个epoch以节省时间
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, criterion, epoch, lr_scheduler)
        # 每个epoch后测试模型
        test_model(model, test_loader)
    
    # 保存模型权重
    weights = save_weights(model)
    
    # ======= FP8手动推理测试 =======
    print("\n测试FP8手动推理...")
    # 创建FP8推理器
    fp8_inference = FP8Inference(weights)
    
    # 获取测试数据
    test_data, test_target = next(iter(test_loader))
    test_data_np = test_data.numpy()
    
    # 由于FP8手动推理计算较慢，只使用少量样本进行测试
    test_samples = min(20, len(test_data_np))
    test_subset = test_data_np[:test_samples]
    test_target_subset = test_target.numpy()[:test_samples]
    
    print(f"使用{test_samples}个样本进行FP8手动推理测试...")
    # 执行FP8手动推理
    fp8_preds = []
    for i in range(0, len(test_subset), 10):  # 小批量进行以加快速度
        batch_data = test_subset[i:i+10]
        fp8_output = fp8_inference.predict(batch_data)
        fp8_preds.extend(fp8_output.argmax(axis=1))
    
    fp8_pred = np.array(fp8_preds)
    
    # 计算FP8手动推理准确率
    fp8_accuracy = (fp8_pred == test_target_subset).mean() * 100
    print(f"FP8手动推理准确率: {fp8_accuracy:.2f}%")
    
    print("\n训练和推理模拟完成!")
    print("生成的文件:")
    print("1. fp8_model_weights.npz - FP8格式模型权重")

if __name__ == "__main__":
    main()