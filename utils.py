import numpy as np

def conv2d(input, kernel, stride=1, padding=0,dtype=np.int32):
    """
    2D卷积操作
    
    Args:
        input: 输入特征图，形状为 (batch_size, in_channels, height, width)
        kernel: 卷积核，形状为 (out_channels, in_channels, kernel_height, kernel_width)
        stride: 步长，默认1
        padding: 填充，默认0
        dtype: 数据类型，默认np.int32
    
    Returns:
        output: 输出特征图，形状为 (batch_size, out_channels, out_height, out_width)
    """
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    # 计算输出特征图的高度和宽度
    out_height = (height - kernel_height + 2 * padding) // stride + 1
    out_width = (width - kernel_width + 2 * padding) // stride + 1
    # 初始化输出特征图
    output = np.zeros((batch_size, out_channels, out_height, out_width), dtype=dtype)
    # 填充输入特征图
    padded_input = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    # 执行卷积操作
    for b in range(batch_size):
        for oc in range(out_channels):
            for h in range(out_height):
                for w in range(out_width):
                    # 计算当前卷积核的位置
                    h_start = h * stride
                    w_start = w * stride
                    # 提取当前卷积核的区域
                    input_region = padded_input[b, :, h_start:h_start+kernel_height, w_start:w_start+kernel_width]
                    # 执行卷积操作
                    output[b, oc, h, w] = np.sum(input_region * kernel[oc])

    return output
