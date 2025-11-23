import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json

class FixedQuantizedCNN(nn.Module):
    def __init__(self):
        super(FixedQuantizedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 14 * 14, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def compute_quantization_params(tensor, num_bits=8, symmetric=True):
    """更稳健的量化参数计算"""
    if symmetric:
        # 对称量化
        max_val = torch.max(torch.abs(tensor)).item()
        scale = max_val / (2**(num_bits-1) - 1)
        zero_point = 0
    else:
        # 非对称量化
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        scale = (max_val - min_val) / (2**num_bits - 1)
        zero_point = round(-min_val / scale)
    
    return scale, zero_point

def quantize_tensor(tensor, scale, zero_point, num_bits=8):
    """量化张量"""
    quantized = torch.clamp(torch.round(tensor / scale + zero_point), 
                           -2**(num_bits-1) if symmetric else 0, 
                           2**(num_bits-1)-1)
    return quantized.to(torch.int8 if num_bits == 8 else torch.int32)

def fixed_quantize_parameters(model, calibration_loader):
    """使用校准数据集计算量化参数"""
    model.eval()
    
    # 收集各层的激活范围
    conv_activations = []
    fc_activations = []
    
    with torch.no_grad():
        for data, _ in calibration_loader:
            # 前向传播并收集激活
            x = model.conv1(data)
            conv_activations.append(x)
            
            x = F.relu(x)
            x = model.pool(x)
            x = x.view(x.size(0), -1)
            x = model.fc(x)
            fc_activations.append(x)
    
    # 计算量化参数
    conv_activation = torch.cat(conv_activations, dim=0)
    fc_activation = torch.cat(fc_activations, dim=0)
    
    # 输入量化参数 (MNIST图像是0-1范围)
    input_scale = 1.0 / 255.0  # 输入已经是0-1，但我们假设原始是0-255
    input_zero_point = 0
    
    # 卷积权重和激活量化参数
    conv_weight_scale, conv_weight_zp = compute_quantization_params(model.conv1.weight.data)
    conv_activation_scale, conv_activation_zp = compute_quantization_params(conv_activation)
    
    # 全连接权重和激活量化参数
    fc_weight_scale, fc_weight_zp = compute_quantization_params(model.fc.weight.data)
    fc_activation_scale, fc_activation_zp = compute_quantization_params(fc_activation)
    
    # 量化权重和偏置
    conv_weight_quantized = quantize_tensor(model.conv1.weight.data, conv_weight_scale, conv_weight_zp)
    
    # 偏置量化 - 使用更高的精度
    # 偏置的scale应该是输入scale * 权重scale
    conv_bias_scale = input_scale * conv_weight_scale
    conv_bias_quantized = quantize_tensor(model.conv1.bias.data, conv_bias_scale, 0, num_bits=32)
    
    fc_weight_quantized = quantize_tensor(model.fc.weight.data, fc_weight_scale, fc_weight_zp)
    fc_bias_scale = conv_activation_scale * fc_weight_scale
    fc_bias_quantized = quantize_tensor(model.fc.bias.data, fc_bias_scale, 0, num_bits=32)
    
    quant_params = {
        'input': {
            'scale': input_scale,
            'zero_point': input_zero_point
        },
        'conv1': {
            'weight': conv_weight_quantized.numpy().tolist(),
            'weight_scale': conv_weight_scale,
            'weight_zero_point': conv_weight_zp,
            'bias': conv_bias_quantized.numpy().tolist(),
            'bias_scale': conv_bias_scale,
            'activation_scale': conv_activation_scale,
            'activation_zero_point': conv_activation_zp
        },
        'fc': {
            'weight': fc_weight_quantized.numpy().tolist(),
            'weight_scale': fc_weight_scale,
            'weight_zero_point': fc_weight_zp,
            'bias': fc_bias_quantized.numpy().tolist(),
            'bias_scale': fc_bias_scale,
            'activation_scale': fc_activation_scale,
            'activation_zero_point': fc_activation_zp
        }
    }
    
    return quant_params

class FixedIntegerInferenceSimulator:
    def __init__(self, quant_params):
        self.quant_params = quant_params
        
        # 加载量化参数
        self.input_scale = quant_params['input']['scale']
        self.input_zero_point = quant_params['input']['zero_point']
        
        self.conv_weight = np.array(quant_params['conv1']['weight'], dtype=np.int8)
        self.conv_weight_scale = quant_params['conv1']['weight_scale']
        self.conv_weight_zp = quant_params['conv1']['weight_zero_point']
        self.conv_bias = np.array(quant_params['conv1']['bias'], dtype=np.int32)
        self.conv_bias_scale = quant_params['conv1']['bias_scale']
        self.conv_activation_scale = quant_params['conv1']['activation_scale']
        self.conv_activation_zp = quant_params['conv1']['activation_zero_point']
        
        self.fc_weight = np.array(quant_params['fc']['weight'], dtype=np.int8)
        self.fc_weight_scale = quant_params['fc']['weight_scale']
        self.fc_weight_zp = quant_params['fc']['weight_zero_point']
        self.fc_bias = np.array(quant_params['fc']['bias'], dtype=np.int32)
        self.fc_bias_scale = quant_params['fc']['bias_scale']
        self.fc_activation_scale = quant_params['fc']['activation_scale']
        self.fc_activation_zp = quant_params['fc']['activation_zero_point']
    
    def fixed_int8_conv2d(self, x_int8, weight_int8, bias_int32, stride=1, padding=1):
        """修复的整数卷积实现"""
        batch_size, in_channels, height, width = x_int8.shape
        out_channels, _, kernel_size, _ = weight_int8.shape
        
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        
        output_int32 = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.int32)
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * stride - padding
                        w_start = ow * stride - padding
                        
                        patch_sum = 0
                        for ic in range(in_channels):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    
                                    if 0 <= h_idx < height and 0 <= w_idx < width:
                                        # 修复：确保在int32中计算
                                        x_val = int(x_int8[b, ic, h_idx, w_idx])
                                        w_val = int(weight_int8[oc, ic, kh, kw])
                                        patch_sum += x_val * w_val
                        
                        output_int32[b, oc, oh, ow] = patch_sum + bias_int32[oc]
        
        return output_int32
    
    def fixed_requantize(self, x_int32, input_scale, weight_scale, input_zp, weight_zp, output_scale, output_zp):
        """修复的重新量化函数"""
        # 更准确的重新量化公式
        # 考虑zero_point的影响
        M = (input_scale * weight_scale) / output_scale
        
        # 应用缩放并添加输出zero_point
        x_int8 = np.clip(np.round(x_int32 * M + output_zp), -128, 127).astype(np.int8)
        return x_int8
    
    def fixed_predict(self, x_float):
        """修复的整数推理流程"""
        # 输入量化
        x_int8 = np.clip(np.round(x_float / self.input_scale + self.input_zero_point), 0, 255).astype(np.int8)
        
        # 卷积层
        conv_output_int32 = self.fixed_int8_conv2d(x_int8, self.conv_weight, self.conv_bias)
        
        # 重新量化到int8
        conv_output_int8 = self.fixed_requantize(
            conv_output_int32, 
            self.input_scale, self.conv_weight_scale,
            self.input_zero_point, self.conv_weight_zp,
            self.conv_activation_scale, self.conv_activation_zp
        )
        
        # ReLU激活 (对于对称量化，ReLU就是clamp到0)
        relu_output_int8 = np.maximum(conv_output_int8, 0)
        
        # 最大池化
        pool_output_int8 = self.int8_max_pool2d(relu_output_int8)
        
        # 展平
        batch_size = pool_output_int8.shape[0]
        flattened = pool_output_int8.reshape(batch_size, -1)
        
        # 全连接层 - 需要修复矩阵乘法
        fc_output_int8 = self.fixed_int8_linear(flattened)
        
        return fc_output_int8
    
    def fixed_int8_linear(self, x_int8):
        """修复的整数全连接层"""
        # 将输入转换为int32进行矩阵乘法
        x_int32 = x_int8.astype(np.int32)
        weight_int32 = self.fc_weight.astype(np.int32)
        
        # 矩阵乘法
        output_int32 = np.dot(x_int32, weight_int32.T) + self.fc_bias
        
        # 重新量化
        output_int8 = self.fixed_requantize(
            output_int32,
            self.conv_activation_scale, self.fc_weight_scale,
            self.conv_activation_zp, self.fc_weight_zp,
            self.fc_activation_scale, self.fc_activation_zp
        )
        
        return output_int8
    
    def int8_max_pool2d(self, x_int8, kernel_size=2, stride=2):
        """整数最大池化实现"""
        batch_size, channels, height, width = x_int8.shape
        
        out_height = height // kernel_size
        out_width = width // kernel_size
        
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.int8)
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * stride
                        w_start = ow * stride
                        h_end = h_start + kernel_size
                        w_end = w_start + kernel_size
                        
                        patch = x_int8[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, oh, ow] = np.max(patch)
        
        return output

def debug_comparison(float_model, simulator, test_loader):
    """调试函数：比较浮点和整数推理的中间结果"""
    float_model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            # 浮点推理
            float_output = float_model(data)
            float_pred = float_output.argmax(dim=1).numpy()
            
            # 整数推理
            data_np = data.numpy()
            int_output = simulator.fixed_predict(data_np)
            int_pred = int_output.argmax(axis=1)
            
            # 比较结果
            accuracy = (int_pred == float_pred).mean()
            print(f"与浮点模型的一致性: {accuracy:.4f}")
            
            if accuracy < 0.5:
                print("警告：整数推理与浮点模型差异很大！")
                # 输出一些调试信息
                print(f"浮点输出范围: [{float_output.min():.4f}, {float_output.max():.4f}]")
                print(f"整数输出范围: [{int_output.min()}, {int_output.max()}]")
            
            break  # 只测试一个batch

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    calibration_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    
    # 创建和训练模型
    print("训练模型...")
    model = FixedQuantizedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 简单训练
    model.train()
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 100:  # 只训练部分数据用于测试
                break
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 测试浮点模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    float_accuracy = correct / total
    print(f"浮点模型准确率: {float_accuracy:.4f}")
    
    # 量化参数
    print("计算量化参数...")
    quant_params = fixed_quantize_parameters(model, calibration_loader)
    
    # 保存参数
    with open('fixed_quantized_params.json', 'w') as f:
        json.dump(quant_params, f, indent=2)
    
    # 创建整数推理模拟器
    simulator = FixedIntegerInferenceSimulator(quant_params)
    
    # 调试比较
    print("调试比较...")
    debug_comparison(model, simulator, test_loader)
    
    # 测试整数推理准确率
    print("测试整数推理...")
    correct = 0
    total = 0
    
    for data, target in test_loader:
        data_np = data.numpy()
        int_output = simulator.fixed_predict(data_np)
        int_pred = int_output.argmax(axis=1)
        correct += (int_pred == target.numpy()).sum()
        total += target.size(0)
    
    int_accuracy = correct / total
    print(f"浮点模型准确率: {float_accuracy:.4f}")
    print(f"整数推理准确率: {int_accuracy:.4f}")
    print(f"准确率损失: {float_accuracy - int_accuracy:.4f}")

if __name__ == "__main__":
    main()