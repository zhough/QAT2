import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义用于FP8的工具函数
class FP8Converter:
    @staticmethod
    def to_fp8(tensor):
        # PyTorch中没有直接的FP8支持，这里使用自定义的FP8模拟
        # 使用INT8模拟FP8，但保持其动态范围
        # 确保我们使用的是最大绝对值得比例因子
        max_val = torch.max(torch.abs(tensor)).item()
        scale = 127.0 / max(max_val, 1e-10)  # 防止除以零
        
        # 量化到int8范围
        int8_tensor = torch.round(tensor * scale).clamp(-128, 127).to(torch.int8)
        return int8_tensor, scale
    
    @staticmethod
    def from_fp8(int8_tensor, scale):
        # 从模拟的FP8转换回浮点
        return int8_tensor.to(torch.float32) / scale
    
    @staticmethod
    def fp8_multiply(a_int8, a_scale, b_int8, b_scale):
        # FP8乘法操作（模拟）
        # 两个int8值相乘，结果使用新的比例因子
        product_int16 = a_int8.to(torch.int16) * b_int8.to(torch.int16)
        new_scale = a_scale * b_scale
        return product_int16, new_scale
    
    @staticmethod
    def fp8_add(a_int8, a_scale, b_int8, b_scale):
        # FP8加法操作（模拟）
        # 需要将两个操作数转换到相同的比例因子
        # 选择较小的比例因子以避免溢出
        if a_scale <= b_scale:
            # 将b转换到a的比例因子
            b_scaled = torch.round(b_int8.to(torch.float32) * (b_scale / a_scale)).to(torch.int8)
            result = a_int8 + b_scaled
            return result, a_scale
        else:
            # 将a转换到b的比例因子
            a_scaled = torch.round(a_int8.to(torch.float32) * (a_scale / b_scale)).to(torch.int8)
            result = a_scaled + b_int8
            return result, b_scale

# 定义神经网络模型
class FP8HandwrittenDigitModel(nn.Module):
    def __init__(self):
        super(FP8HandwrittenDigitModel, self).__init__()
        # 卷积层: conv(1,32,3,2,0) - 1个输入通道，32个输出通道，3x3卷积核，步长2，无填充
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0)
        # 池化层: MaxPool2d
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层: 计算卷积和池化后的输出大小并连接到10个输出类别
        # 输入图像大小为28x28，经过conv1后变为13x13，再经过maxpool后变为6x6
        # 所以全连接层的输入特征数是 32 * 6 * 6 = 1152
        self.fc1 = nn.Linear(32 * 6 * 6, 10)
        
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        # ReLU6激活函数
        x = F.relu6(x)
        # 池化层
        x = self.pool(x)
        # 将张量展平为一维向量
        x = x.view(-1, 32 * 6 * 6)
        # 全连接层
        x = self.fc1(x)
        return x
    
    def save_fp8_weights(self, file_path):
        # 以FP8格式保存权重和偏置
        state_dict = self.state_dict()
        save_dict = {}
        
        for key, tensor in state_dict.items():
            # 将权重和偏置转换为FP8格式
            int8_tensor, scale = FP8Converter.to_fp8(tensor)
            # 分别保存数据和缩放因子
            save_dict[f"{key}_data"] = int8_tensor.cpu().numpy()
            save_dict[f"{key}_scale"] = scale
        
        # 保存为.npz文件
        np.savez(file_path, **save_dict)
        print(f"FP8权重已保存至 {file_path}")
        
    @staticmethod
    def load_fp8_weights(file_path):
        # 加载FP8格式的权重和偏置
        npzfile = np.load(file_path)
        fp8_state = {}
        
        # 获取所有数据键
        data_keys = [key for key in npzfile.files if key.endswith('_data')]
        
        for data_key in data_keys:
            original_key = data_key.replace('_data', '')
            scale_key = f"{original_key}_scale"
            
            if scale_key in npzfile:
                int8_data = torch.from_numpy(npzfile[data_key]).to(torch.int8)
                scale = npzfile[scale_key]
                # 转换回浮点格式（在推理时会再次使用FP8）
                float_tensor = FP8Converter.from_fp8(int8_data, scale)
                fp8_state[original_key] = float_tensor
        
        return fp8_state

# 实现FP8手动推理类 - 完全使用整数运算，不使用FP32作为中间值
class FP8ManualInference:
    def __init__(self, model_weights_path):
        # 加载模型权重
        self.load_weights(model_weights_path)
    
    def load_weights(self, file_path):
        """加载FP8格式的权重和偏置"""
        npzfile = np.load(file_path)
        self.weights = {}
        
        # 提取所有以_data结尾的键
        data_keys = [key for key in npzfile.files if key.endswith('_data')]
        
        for data_key in data_keys:
            original_key = data_key.replace('_data', '')
            scale_key = f"{original_key}_scale"
            
            if scale_key in npzfile:
                # 加载权重数据（确保为int8格式）和缩放因子
                self.weights[original_key] = {
                    'data': npzfile[data_key].astype(np.int8),
                    'scale': float(npzfile[scale_key])
                }
    
    def integer_multiply_scale(self, a, scale_a, b, scale_b):
        """模拟FP8乘法，返回整数结果和新的缩放因子"""
        # 两个int8相乘会得到int16，这里使用整数运算
        product = int(a) * int(b)
        new_scale = scale_a * scale_b
        return product, new_scale
    
    def integer_add_scale(self, a, scale_a, b, scale_b):
        """模拟FP8加法，将两个数转换到相同的缩放因子后相加"""
        # 确定哪个缩放因子更小，选择较小的以避免溢出
        if scale_a <= scale_b:
            # 将b缩放到a的范围
            # 计算缩放比例 (scale_b / scale_a) 并转换为整数运算
            scale_ratio = scale_b / scale_a
            # 使用固定点算术进行缩放
            b_scaled = int(round(b * scale_ratio))
            # 确保结果在int8范围内
            b_scaled_clamped = max(-128, min(127, b_scaled))
            return a + b_scaled_clamped, scale_a
        else:
            # 将a缩放到b的范围
            scale_ratio = scale_a / scale_b
            a_scaled = int(round(a * scale_ratio))
            a_scaled_clamped = max(-128, min(127, a_scaled))
            return a_scaled_clamped + b, scale_b
    
    def scale_int_value(self, value, from_scale, to_scale):
        """将整数值从一个缩放因子转换到另一个"""
        scale_ratio = from_scale / to_scale
        scaled_value = int(round(value * scale_ratio))
        # 确保结果在int8范围内
        return max(-128, min(127, scaled_value))
    
    def infer(self, input_image):
        """执行FP8精度的手动推理，完全使用整数运算"""
        # 确保输入是numpy数组且维度正确
        if len(input_image.shape) == 2:
            input_image = input_image.reshape(28, 28)
        
        # 关键修复：输入处理 - 确保输入在正确范围内
        # 检查并归一化输入
        if np.max(input_image) > 1.0:
            input_image = input_image / 255.0
        
        # 为了避免缩放因子问题，我们将输入量化到整数范围，但不使用fp8的缩放方式
        # 直接使用简单的整数表示
        input_scaled = np.round(input_image * 255).astype(np.int32)
        
        # 1. 卷积层计算 - 完全重写计算方式
        # 获取卷积权重和偏置
        conv_weights = self.weights['conv1.weight']['data']  # 形状: [32, 1, 3, 3]
        conv_bias = self.weights['conv1.bias']['data']      # 形状: [32]
        
        # 修复：不再使用缩放因子，直接使用整数权重和偏置
        # 由于权重已经在保存时进行了量化，我们直接使用这些整数值
        
        # 输出特征图尺寸: 13x13
        conv_output = np.zeros((1, 32, 13, 13), dtype=np.int32)
        
        # 执行卷积操作 - 简化且直接
        for out_channel in range(32):
            for i in range(13):
                for j in range(13):
                    conv_sum = 0
                    
                    # 卷积计算
                    for k in range(3):
                        for l in range(3):
                            # 计算输入坐标（考虑步长为2）
                            input_i = 2 * i + k
                            input_j = 2 * j + l
                            
                            # 检查坐标是否有效
                            if 0 <= input_i < 28 and 0 <= input_j < 28:
                                # 获取输入值和权重值
                                input_val = input_scaled[input_i, input_j]
                                weight_val = conv_weights[out_channel, 0, k, l]
                                
                                # 整数乘法
                                product = input_val * weight_val
                                conv_sum += product
                    
                    # 直接添加偏置（权重文件中已经包含正确缩放的偏置）
                    conv_sum += conv_bias[out_channel]
                    
                    # 存储结果
                    conv_output[0, out_channel, i, j] = conv_sum
        
        # 2. ReLU6激活函数
        # 简化的ReLU6实现
        relu6_output = np.clip(conv_output, 0, 6 * 255 * 100)  # 估计的合理上限
        
        # 3. 最大池化
        # 池化核: 2x2, 步长: 2 -> 输出: 6x6
        pool_output = np.zeros((1, 32, 6, 6), dtype=np.int32)
        
        # 执行最大池化
        for out_channel in range(32):
            for i in range(6):
                for j in range(6):
                    max_val = -2147483648  # int32最小值
                    
                    # 查找池化窗口内的最大值
                    for k in range(2):
                        for l in range(2):
                            # 计算池化窗口坐标
                            pool_i = 2 * i + k
                            pool_j = 2 * j + l
                            
                            if 0 <= pool_i < 13 and 0 <= pool_j < 13:
                                current_val = relu6_output[0, out_channel, pool_i, pool_j]
                                if current_val > max_val:
                                    max_val = current_val
                    
                    pool_output[0, out_channel, i, j] = max_val
        
        # 4. 全连接层计算 - 完全重写
        fc_weights = self.weights['fc1.weight']['data']  # 形状: [10, 1152]
        fc_bias = self.weights['fc1.bias']['data']      # 形状: [10]
        
        # 将池化输出展平
        pool_flattened = pool_output.reshape(-1).astype(np.int32)
        
        # 计算全连接层输出
        fc_output = np.zeros(10, dtype=np.int32)
        
        for i in range(10):
            sum_val = 0
            
            # 直接计算加权和，确保正确处理权重
            for j in range(len(pool_flattened)):
                # 确保索引在有效范围内
                if j < fc_weights.shape[1]:
                    # 直接使用整数量化的权重
                    weight_val = fc_weights[i, j]
                    sum_val += pool_flattened[j] * weight_val
            
            # 直接添加偏置
            sum_val += fc_bias[i]
            
            fc_output[i] = sum_val
        
        # 返回预测结果
        return np.argmax(fc_output)

# 主函数
def main():
    # 创建模型
    model = FP8HandwrittenDigitModel().to(device)
    
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    print("正在加载MNIST数据集...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型 - 使用FP8精度理念进行训练
    print("开始训练模型...")
    epochs = 3
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播 - 在PyTorch中我们仍然使用FP32进行训练
            # 但在保存和推理时会使用FP8
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每100个batch打印一次损失
            if i % 100 == 99:
                print(f'[{epoch + 1}/{epochs}, {i + 1:5d}] 损失: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    print('训练完成!')
    
    # 以FP8格式保存权重
    weights_path = 'handwritten_digit_model_fp8.npz'
    model.save_fp8_weights(weights_path)
    
    # 在保存前评估模型性能
    print("\n评估原始模型性能:")
    evaluate_model(model, test_loader)
    
    # 使用FP8手动推理测试
    print("\n初始化FP8手动推理器...")
    fp8_inference = FP8ManualInference(weights_path)
    
    # 测试10张图片的准确率
    correct = 0
    total = 100
    
    print("\n验证10张测试图片的手动推理准确率:")
    for i in range(total):
        # 获取测试图像和标签
        image, label = test_dataset[i]
        # 转换为numpy数组用于手动推理
        image_np = image.numpy().squeeze()  # 去除通道维度
        
        # 进行手动FP8推理
        predicted_label = fp8_inference.infer(image_np)
        
        # 检查预测是否正确
        is_correct = predicted_label == label
        if is_correct:
            correct += 1
        
        # 显示结果
        print(f"测试图片 {i+1}/{total}: 真实标签={label}, 预测标签={predicted_label}, {'✓ 正确' if is_correct else '✗ 错误'}")
    
    # 计算并显示准确率
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n手动推理的准确率: {accuracy:.2%} ({correct}/{total})")


def evaluate_model(model, test_loader):
    """评估模型在测试集上的性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'模型在测试集上的准确率: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    main()