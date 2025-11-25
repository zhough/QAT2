import numpy as np
import os

def load_and_analyze_weights(weights_path='handwritten_digit_model_fp8.npz'):
    data = np.load(weights_path)
    # 打印所有可用的权重键
    print(f"\n权重文件包含的键: {list(data.keys())}")
    weights_info = {}
    # 分析每个权重数组
    for key in data:
        weight_array = data[key]
        weights_info[key] = {
            'shape': weight_array.shape,
            'dtype': str(weight_array.dtype),
            'min_value': weight_array.min(),
            'max_value': weight_array.max(),
            'unique_values': list(set(weight_array.flatten()))
        }
        
        # 打印权重信息
        print(f"\n--- {key} ---")
        print(f"  形状: {weight_array.shape}")
        print(f"  数据类型: {weight_array.dtype}")
        print(f"  最小值: {weight_array.min()}")
        print(f"  最大值: {weight_array.max()}")
        print(f"  唯一值数量: {len(set(weight_array.flatten()))}")
        
        # 如果是卷积权重，分析其结构
        if key.startswith('conv1') and 'weight' in key and len(weight_array.shape) >= 4:
            print(f"  卷积核数量: {weight_array.shape[0]}")
            print(f"  输入通道数: {weight_array.shape[1]}")
            print(f"  卷积核大小: {weight_array.shape[2]}x{weight_array.shape[3]}")
    
    # 检查是否存在缩放因子
    scale_keys = [key for key in data if 'scale' in key.lower()]
    if scale_keys:
        print(f"\n找到缩放因子: {scale_keys}")
        for key in scale_keys:
            print(f"  {key}: {data[key]}")
    
    return weights_info

def extract_weights_for_inference(weights_path='handwritten_digit_model_fp8.npz', output_path=None):
    """
    提取权重并准备用于推理
    
    Args:
        weights_path: 权重文件路径
        output_path: 输出路径，如果为None则不保存
    
    Returns:
        dict: 用于推理的权重字典
    """
    print(f"\n正在提取用于推理的权重...")
    data = np.load(weights_path)
    
    # 构建推理权重字典
    inference_weights = {}
    
    # 提取所有权重和缩放因子
    for key in data:
        inference_weights[key] = data[key].copy()
    
    if output_path:
        np.savez(output_path, **inference_weights)
        print(f"推理权重已保存至: {output_path}")
    
    return inference_weights

def main():
    """
    主函数，加载并分析权重
    """
    print("=== FP8手写数字识别模型权重分析工具 ===")
    
    # 加载并分析权重
    weights_info = load_and_analyze_weights()
    
    if weights_info:
        # 提取推理权重
        extract_weights_for_inference()
        
        print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()