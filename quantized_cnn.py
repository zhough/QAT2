import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json

# 定义量化感知训练模型
class QuantizedCNN(nn.Module):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        # 量化stub
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 网络层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 14 * 14, 10)
        
        # 量化配置
        self.quant_config = {}
        
    def forward(self, x):
        x = self.quant(x)
        
        # 卷积层
        x = self.conv1(x)
        # 手动记录卷积的量化参数
        if not self.quant_config:
            self._record_conv_quant_params(x)
        
        x = self.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc(x)
        
        x = self.dequant(x)
        return x
    
    def _record_conv_quant_params(self, conv_output):
        """记录卷积层的量化参数"""
        # 计算输出的scale和zero_point
        min_val = conv_output.min().item()
        max_val = conv_output.max().item()
        
        # 对称量化参数
        scale = max(abs(min_val), abs(max_val)) / 127.0
        zero_point = 0  # 对称量化zero_point为0
        
        self.quant_config['conv_output'] = {
            'scale': scale,
            'zero_point': zero_point
        }

# 训练函数
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 测试函数
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# 量化权重和偏置
def quantize_parameters(model):
    """量化模型参数并保存量化信息"""
    quant_params = {}
    
    # 量化卷积层权重
    conv_weight = model.conv1.weight.data
    conv_weight_min = conv_weight.min().item()
    conv_weight_max = conv_weight.max().item()
    conv_weight_scale = max(abs(conv_weight_min), abs(conv_weight_max)) / 127.0
    
    # 量化到int8
    conv_weight_quantized = torch.clamp(torch.round(conv_weight / conv_weight_scale), -128, 127).to(torch.int8)
    
    # 量化卷积层偏置 (使用更高的位宽int32)
    conv_bias = model.conv1.bias.data
    # 偏置的scale是权重scale和输入scale的乘积
    input_scale = 1.0 / 255.0  # 输入图像是0-255，归一化到0-1
    conv_bias_scale = conv_weight_scale * input_scale
    conv_bias_quantized = torch.round(conv_bias / conv_bias_scale).to(torch.int32)
    
    quant_params['conv1'] = {
        'weight': conv_weight_quantized.numpy().tolist(),
        'weight_scale': conv_weight_scale,
        'bias': conv_bias_quantized.numpy().tolist(),
        'bias_scale': conv_bias_scale
    }
    
    # 量化全连接层权重
    fc_weight = model.fc.weight.data
    fc_weight_min = fc_weight.min().item()
    fc_weight_max = fc_weight.max().item()
    fc_weight_scale = max(abs(fc_weight_min), abs(fc_weight_max)) / 127.0
    
    fc_weight_quantized = torch.clamp(torch.round(fc_weight / fc_weight_scale), -128, 127).to(torch.int8)
    
    # 量化全连接层偏置
    fc_bias = model.fc.bias.data
    fc_bias_scale = fc_weight_scale * model.quant_config['conv_output']['scale']
    fc_bias_quantized = torch.round(fc_bias / fc_bias_scale).to(torch.int32)
    
    quant_params['fc'] = {
        'weight': fc_weight_quantized.numpy().tolist(),
        'weight_scale': fc_weight_scale,
        'bias': fc_bias_quantized.numpy().tolist(),
        'bias_scale': fc_bias_scale
    }
    
    # 保存激活函数的量化参数
    quant_params['activation'] = model.quant_config['conv_output']
    quant_params['input_scale'] = input_scale
    
    return quant_params

# 整数推理模拟
class IntegerInferenceSimulator:
    def __init__(self, quant_params):
        self.quant_params = quant_params
        
        # 加载量化后的参数
        self.conv_weight = np.array(quant_params['conv1']['weight'], dtype=np.int8)
        self.conv_weight_scale = quant_params['conv1']['weight_scale']
        self.conv_bias = np.array(quant_params['conv1']['bias'], dtype=np.int32)
        self.conv_bias_scale = quant_params['conv1']['bias_scale']
        
        self.fc_weight = np.array(quant_params['fc']['weight'], dtype=np.int8)
        self.fc_weight_scale = quant_params['fc']['weight_scale']
        self.fc_bias = np.array(quant_params['fc']['bias'], dtype=np.int32)
        self.fc_bias_scale = quant_params['fc']['bias_scale']
        
        self.activation_scale = quant_params['activation']['scale']
        self.input_scale = quant_params['input_scale']
    
    def int8_conv2d(self, x_int8, weight_int8, bias_int32, stride=1, padding=1):
        """整数卷积实现 - 修复溢出问题"""
        batch_size, in_channels, height, width = x_int8.shape
        out_channels, _, kernel_size, _ = weight_int8.shape
        
        # 输出尺寸计算
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        
        # 初始化输出 (使用int32累积)
        output_int32 = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.int32)
        
        # 实现卷积操作
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # 计算感受野位置
                        h_start = oh * stride - padding
                        w_start = ow * stride - padding
                        h_end = h_start + kernel_size
                        w_end = w_start + kernel_size
                        
                        # 提取patch并进行卷积 - 修复：确保使用int32计算
                        patch_sum = 0
                        for ic in range(in_channels):
                            for kh in range(kernel_size):
                                for kw in range(kernel_size):
                                    h_idx = h_start + kh
                                    w_idx = w_start + kw
                                    
                                    # 处理边界填充 (零填充)
                                    if (0 <= h_idx < height and 0 <= w_idx < width):
                                        # 关键修复：将int8转换为int32再进行乘法
                                        x_val = np.int32(x_int8[b, ic, h_idx, w_idx])
                                        w_val = np.int32(weight_int8[oc, ic, kh, kw])
                                        patch_sum += x_val * w_val
                        
                        # 添加偏置
                        output_int32[b, oc, oh, ow] = patch_sum + bias_int32[oc]
        
        return output_int32
    
    def requantize_int32_to_int8(self, x_int32, input_scale, weight_scale, output_scale):
        """将int32转换回int8"""
        # 计算总的scale因子
        total_scale = (input_scale * weight_scale) / output_scale
        
        # 缩放并转换为int8
        x_int8 = np.clip(np.round(x_int32 * total_scale), -128, 127).astype(np.int8)
        return x_int8
    
    def int8_relu(self, x_int8):
        """整数ReLU实现"""
        return np.maximum(x_int8, 0)
    
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
    
    def int8_linear(self, x_int8, weight_int8, bias_int32, input_scale, weight_scale, output_scale):
        """整数全连接层实现 - 修复溢出问题"""
        # 修复：将int8转换为int32再进行矩阵乘法
        x_int32 = x_int8.astype(np.int32)
        weight_int32 = weight_int8.astype(np.int32)
        
        # 矩阵乘法 (int32 * int32 -> int32累积)
        output_int32 = np.dot(x_int32, weight_int32.T) + bias_int32
        
        # 重新量化到int8
        total_scale = (input_scale * weight_scale) / output_scale
        output_int8 = np.clip(np.round(output_int32 * total_scale), -128, 127).astype(np.int8)
        
        return output_int8
    
    def predict(self, x_float):
        """完整的整数推理流程"""
        # 输入量化到int8
        x_int8 = np.clip(np.round(x_float / self.input_scale), 0, 255).astype(np.int8)
        
        # 卷积层 (int8 * int8 -> int32 -> int8)
        conv_output_int32 = self.int8_conv2d(x_int8, self.conv_weight, self.conv_bias)
        conv_output_int8 = self.requantize_int32_to_int8(
            conv_output_int32, self.input_scale, self.conv_weight_scale, self.activation_scale
        )
        
        # ReLU激活
        relu_output_int8 = self.int8_relu(conv_output_int8)
        
        # 最大池化
        pool_output_int8 = self.int8_max_pool2d(relu_output_int8)
        
        # 展平
        batch_size = pool_output_int8.shape[0]
        flattened = pool_output_int8.reshape(batch_size, -1)
        
        # 全连接层
        fc_output_int8 = self.int8_linear(
            flattened, self.fc_weight, self.fc_bias,
            self.activation_scale, self.fc_weight_scale, self.activation_scale
        )
        
        # 对于输出层，我们直接返回int32结果用于分类
        return fc_output_int8

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建模型
    model = QuantizedCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("开始训练模型...")
    for epoch in range(1, 3):  # 训练2个epoch
        train_model(model, train_loader, optimizer, criterion, epoch)
        test_model(model, test_loader)
    
    # 量化参数
    print("量化模型参数...")
    quant_params = quantize_parameters(model)
    
    # 保存量化参数
    with open('quantized_params.json', 'w') as f:
        json.dump(quant_params, f, indent=2)
    print("量化参数已保存到 quantized_params.json")
    
    # 创建整数推理模拟器
    simulator = IntegerInferenceSimulator(quant_params)
    
    # 测试整数推理
    print("测试整数推理...")
    test_data, test_target = next(iter(test_loader))
    test_data_np = test_data.numpy()
    
    # 浮点模型推理
    model.eval()
    with torch.no_grad():
        float_output = model(test_data)
        float_pred = float_output.argmax(dim=1).numpy()
    
    # 整数推理 - 使用小批量避免内存问题
    batch_size = 100
    int_preds = []
    
    for i in range(0, len(test_data_np), batch_size):
        batch_data = test_data_np[i:i+batch_size]
        int_output = simulator.predict(batch_data)
        int_preds.extend(int_output.argmax(axis=1))
    
    int_pred = np.array(int_preds)
    
    # 计算准确率
    float_accuracy = (float_pred == test_target.numpy()).mean()
    int_accuracy = (int_pred == test_target.numpy()).mean()
    
    print(f"浮点模型准确率: {float_accuracy:.4f}")
    print(f"整数推理准确率: {int_accuracy:.4f}")
    print(f"准确率损失: {float_accuracy - int_accuracy:.4f}")
    
    # 打印量化参数信息
    print("\n量化参数信息:")
    print(f"卷积权重 scale: {quant_params['conv1']['weight_scale']:.6f}")
    print(f"卷积偏置 scale: {quant_params['conv1']['bias_scale']:.6f}")
    print(f"全连接权重 scale: {quant_params['fc']['weight_scale']:.6f}")
    print(f"全连接偏置 scale: {quant_params['fc']['bias_scale']:.6f}")
    print(f"激活输出 scale: {quant_params['activation']['scale']:.6f}")

if __name__ == "__main__":
    main()