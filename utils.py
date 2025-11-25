import re
import numpy as np
import struct
import os

def conv2d(input, kernel,bias=None, stride=1, padding=0,dtype=np.int32):
    """
    2D卷积操作
    Args:
        input: 输入特征图，形状为 (in_channels, height, width)
        kernel: 卷积核，形状为 (out_channels, in_channels, kernel_height, kernel_width)
        bias: 偏置项，形状为 (out_channels,)，默认None
        stride: 步长，默认1
        padding: 填充，默认0
        dtype: 数据类型，默认np.int32
    
    Returns:
        output: 输出特征图，形状为 (out_channels, out_height, out_width)
    """
    in_channels, height, width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    # 计算输出特征图的高度和宽度
    out_height = (height - kernel_height + 2 * padding) // stride + 1
    out_width = (width - kernel_width + 2 * padding) // stride + 1
    # 初始化输出特征图
    output = np.zeros((out_channels, out_height, out_width), dtype=dtype)
    # 填充输入特征图
    padded_input = np.pad(input, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    # 执行卷积操作
    for oc in range(out_channels):
        for h in range(out_height):
            for w in range(out_width):
                # 计算当前卷积核的位置
                h_start = h * stride
                w_start = w * stride
                # 提取当前卷积核的区域
                input_region = padded_input[:, h_start:h_start+kernel_height, w_start:w_start+kernel_width]
                # 执行卷积操作
                output[oc, h, w] = np.sum(input_region * kernel[oc])
                # 添加偏置项
                if bias is not None:
                    output[oc, h, w] += bias[oc]
    
    return output


def fully_connected(input, weight, bias=None,dtype=np.int32):
    """
    全连接层操作
    
    Args:
        input: 输入特征图，形状为 (1，in_features)
        weight: 权重矩阵，形状为 (out_features, in_features)
        bias: 偏置项，形状为 (out_features,)，默认None
        dtype: 数据类型，默认np.int32
    
    Returns:
        output: 输出特征图，形状为 (out_features)
    """
    in_features = input.shape[0]
    out_features, _ = weight.shape
    # 初始化输出特征图
    output = np.zeros((1, out_features), dtype=dtype)
    # 执行全连接操作
    for oc in range(out_features):
        # 计算当前输出特征图的元素，确保结果是标量
        dot_product = np.dot(input, weight[oc])
        output[0, oc] = dot_product.item() if hasattr(dot_product, 'item') else dot_product
        # 添加偏置项
        if bias is not None:
            output[0, oc] += bias[oc]
    
    return output[0]

def pooling(input,kernel_size=2,stride=2):
    """
    池化层操作
    
    Args:
        input: 输入特征图，形状为 (in_channels, height, width)
        kernel_size: 池化核大小，默认2
        stride: 步长，默认2
    
    Returns:
        output: 输出特征图，形状为 (in_channels, out_height, out_width)
    """
    in_channels, height, width = input.shape
    # 计算输出特征图的高度和宽度
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    # 初始化输出特征图
    output = np.zeros((in_channels, out_height, out_width), dtype=input.dtype)
    # 执行池化操作
    for ic in range(in_channels):
        for h in range(out_height):
            for w in range(out_width):
                # 计算当前池化核的位置
                h_start = h * stride
                w_start = w * stride
                # 提取当前池化核的区域
                input_region = input[ic, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                # 执行池化操作
                output[ic, h, w] = np.max(input_region)
    return output

def reluk(input,k=1,dtype=np.int32):
    """
    ReLU-K 激活函数
    
    Args:
        input: 输入特征图，形状为 (in_channels, height, width)
        k: 饱和参数
    Returns:
        output: 输出特征图，形状为 (in_channels, height, width)
    """
    return np.clip(input, 0, k).astype(dtype)

def load_weight(file_path):
    file = np.load(file_path)
    return file

def write_weight_to_binary(file_path):
    #'handwritten_digit_model_fp8.npz'
    npfiles = load_weight(file_path)
    data_keys = [key for key in npfiles.files if key.endswith('_data')]
    print(data_keys)
    for key in data_keys:
        print(npfiles[key].shape)


def load_weight_from_npz(file_path):
    npfiles = load_weight(file_path)
    data_keys = [key for key in npfiles.files if key.endswith('_data')]
    conv1_weight = npfiles['conv1.weight_data']
    conv1_bias = npfiles['conv1.bias_data']
    fc1_weight = npfiles['fc1.weight_data']
    fc1_bias = npfiles['fc1.bias_data']
    return conv1_weight, conv1_bias, fc1_weight, fc1_bias

def read_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        # 解析文件头
        magic, num_images, num_rows, num_cols = struct.unpack('>iiii', data[:16])
        # 解析图像数据
        images = np.frombuffer(data[16:], dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        return images

def read_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        # 解析文件头
        magic, num_labels = struct.unpack('>ii', data[:8])
        # 解析标签数据
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        return labels


def infer(input,conv1_weight,conv1_bias,fc1_weight,fc1_bias):
    #量化输入
    input = (input/(255/3)).astype(np.int16)
    # 卷积层
    conv_output = conv2d(input, conv1_weight, conv1_bias, stride=2, padding=0,dtype=np.int16)
    # 池化层
    pool_output = pooling(conv_output,kernel_size=2,stride=2)
    # ReLU-K 激活函数
    reluk_output = reluk(pool_output,1,dtype=np.int16)
    # 展平特征图
    reluk_output = reluk_output.reshape(1, -1)
    # 全连接层
    fc_output = fully_connected(reluk_output, fc1_weight, fc1_bias,dtype=np.int16)
    return fc_output,np.argmax(fc_output)


def eval(img_num):
    correct = 0
    images = read_mnist_images('data/MNIST/raw/t10k-images-idx3-ubyte')
    labels = read_mnist_labels('data/MNIST/raw/t10k-labels-idx1-ubyte')
    conv1_weight, conv1_bias, fc1_weight, fc1_bias = load_weight_from_npz('handwritten_digit_model_fp8.npz')
    for i in range(img_num):
        output, pred = infer(images[i][np.newaxis, ...], conv1_weight, conv1_bias, fc1_weight, fc1_bias)
        if pred == labels[i]:
            correct += 1
    return correct/img_num



if __name__ == '__main__':
    p = eval(100)
    print(p)










