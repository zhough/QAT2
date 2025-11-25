from torch import relu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

OUT_CHANNEL = 16
def to_int8(tensor):
    # PyTorch中没有直接的FP8支持，这里使用自定义的FP8模拟
    # 使用INT8模拟FP8，但保持其动态范围
    # 确保我们使用的是最大绝对值得比例因子
    max_val = torch.max(torch.abs(tensor)).item()
    scale = 127.0 / max(max_val, 1e-10)  # 防止除以零
    
    # 量化到int8范围
    int8_tensor = torch.round(tensor * scale).clamp(-128, 127).to(torch.int8)
    return int8_tensor, scale
    


# 定义神经网络模型
class FP8HandwrittenDigitModel(nn.Module):
    def __init__(self):
        super(FP8HandwrittenDigitModel, self).__init__()
        self.conv1 = nn.Conv2d(1, OUT_CHANNEL, kernel_size=3, stride=2, padding=0)
        # 池化层: MaxPool2d
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_size = OUT_CHANNEL * 6 * 6
        self.fc1 = nn.Linear(self.input_size, 10)

    def relu3(self,x):
        return torch.clamp(x,min=0,max=3)
    def relu6(self,x):
        return torch.clamp(x,min=0,max=6)
    def relu1(self,x):
        return torch.clamp(x,min=0,max=1)
    def relu05(self,x):
        return torch.clamp(x,min=0,max=0.5)
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        # ReLU6激活函数
        x = self.relu05(x)
        # 池化层
        x = self.pool(x)
        # 将张量展平为一维向量
        x = x.view(-1, self.input_size)
        # 全连接层
        x = self.fc1(x)
        return x
    
    def save_int8_weights(self, file_path):
        # 以INT8格式保存权重和偏置
        state_dict = self.state_dict()
        save_dict = {}
        
        for key, tensor in state_dict.items():
            # 将权重和偏置转换为INT8格式
            int8_tensor, scale = to_int8(tensor)
            # 分别保存数据和缩放因子
            save_dict[f"{key}_data"] = int8_tensor.cpu().numpy()
            save_dict[f"{key}_scale"] = scale
        
        # 保存为.npz文件
        np.savez(file_path, **save_dict)
        print(f"FP8权重已保存至 {file_path}")
        


class FP8ManualInference:
    def __init__(self, model_weights_path):
        # 加载模型权重
        self.load_weights(model_weights_path)
    
    def load_weights(self, file_path):
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
    

    
    def infer(self, input_image,infer_type=np.int16,output_type=np.int16):
        """执行整数运算"""
        # 确保输入是numpy数组且维度正确
        if len(input_image.shape) == 2:
            input_image = input_image.reshape(28, 28)
        

        # if np.max(input_image) > 1.0:
        #     input_image = input_image / 255.0
        # input_scaled = np.round(input_image * 255).astype(infer_type)
        #input_scaled = np.round(input_image).astype(infer_type)
        input_scaled = input_image.astype(infer_type)
        #print(f'input_scaled取值: {list(set(input_scaled.flatten()))}')
        # 1. 卷积层计算 - 完全重写计算方式
        # 获取卷积权重和偏置
        conv_weights = self.weights['conv1.weight']['data']  # 形状: [32, 1, 3, 3]
        conv_bias = self.weights['conv1.bias']['data']      # 形状: [32]
        
        # 修复：不再使用缩放因子，直接使用整数权重和偏置
        # 由于权重已经在保存时进行了量化，我们直接使用这些整数值
        
        # 输出特征图尺寸: 13x13
        conv_output = np.zeros((1, OUT_CHANNEL, 13, 13), dtype=infer_type)
        
        # 执行卷积操作 - 简化且直接
        for out_channel in range(OUT_CHANNEL):
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
        print(f'conv_output最大值：{np.max(conv_output)}')

        relu6_output = np.clip(conv_output, 0, 1)
        #print(f'relu6_output{relu6_output}') 
        print(f'relu6_output最大值：{np.max(relu6_output)}')
        # 3. 最大池化
        # 池化核: 2x2, 步长: 2 -> 输出: 6x6
        pool_output = np.zeros((1, OUT_CHANNEL, 6, 6), dtype=infer_type)
        # 执行最大池化
        for out_channel in range(OUT_CHANNEL):
            for i in range(6):
                for j in range(6):
                    max_val = -2**31 if infer_type == np.int32 else -2**15
                    
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
        pool_flattened = pool_output.reshape(-1).astype(output_type)
        print(f'pool_output最大值：{np.max(pool_output)}')
        # 计算全连接层输出
        fc_output = np.zeros(10, dtype=output_type)
        
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
        print(f'fc_output最大值：{np.max(fc_output)}')
        # 返回预测结果
        return np.argmax(fc_output)

# 主函数
def main(train=True,test_num=100):
    # 创建模型
    model = FP8HandwrittenDigitModel().to(device)
    
    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1/3,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    print("正在加载MNIST数据集...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    infer_dataset1 = datasets.MNIST('./data', train=False)
    infer_dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64*4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    if train:
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
        model.save_int8_weights(weights_path)
        
        print("\n评估原始模型性能:")
        evaluate_model(model, test_loader)
    
    # 使用FP8手动推理测试
    print("\n初始化模拟推理器...")
    weights_path = 'handwritten_digit_model_fp8.npz'
    fp8_inference = FP8ManualInference(weights_path)
    #推理1
    correct = 0
    total = test_num
    print(f"\n验证{total}张测试图片的手动推理准确率:")
    for i in range(total):
        # 获取测试图像和标签
        image, label = infer_dataset1[i]
        # 转换为numpy数组用于手动推理
        # image_np = np.array(image, dtype=np.float32) / 255.0
        # image_np = image_np*3
        image_np = np.array(image,dtype=np.uint8)/(255/3-1)
        # 进行手动FP8推理
        predicted_label = fp8_inference.infer(image_np)
        # 检查预测是否正确
        is_correct = predicted_label == label
        if is_correct:
            correct += 1
        # 显示结果
        #print(f"测试图片 {i+1}/{total}: 真实标签={label}, 预测标签={predicted_label}, {'✓ 正确' if is_correct else '✗ 错误'}")
    # 计算并显示准确率
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n手动推理1的准确率: {accuracy:.2%} ({correct}/{total})")



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
    main(train=False,test_num=100)
