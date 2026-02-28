# -*- coding: utf-8 -*-
"""
单张眼底图像糖尿病视网膜病变分级推理脚本
使用训练好的 ImprovedResNetDR 模型对图像进行分类
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
IMG_SIZE = 384
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称（与训练时保持一致）
CLASS_NAMES = ['无病变', '轻度', '中度', '重度', '增殖性']

# 模型权重路径（请根据实际路径修改）
MODEL_PATH = r"F:\DRDiaSys\best_resnet_aptos_enhanced.pth"


# ==================== 模型定义 ====================
class ImprovedResNetDR(nn.Module):
    """糖尿病视网膜病变分类模型（与训练脚本保持一致）"""
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.5):
        super(ImprovedResNetDR, self).__init__()
        
        # 使用ResNet34作为骨干网络
        self.backbone = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # 获取特征维度
        num_features = self.backbone.fc.in_features
        
        # 分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ==================== 数据预处理 ====================
def get_inference_transforms():
    """推理时的数据预处理（与训练时验证集保持一致）"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def preprocess_image(image_path, transform):
    """
    加载并预处理单张图像
    
    Args:
        image_path: 图像文件路径
        transform: albumentations变换
    
    Returns:
        tensor: 预处理后的图像张量
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用预处理
    if transform:
        transformed = transform(image=image)
        image = transformed['image']
    
    return image


# ==================== 模型加载 ====================
def load_model(model_path, device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        device: 计算设备
    
    Returns:
        model: 加载了权重的模型
    """
    print(f"🔄 正在加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型
    model = ImprovedResNetDR(num_classes=NUM_CLASSES, pretrained=False)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功，使用设备: {device}")
    return model


# ==================== 推理函数 ====================
def predict_single_image(model, image_tensor, device):
    """
    对单张图像进行推理
    
    Args:
        model: 已加载的模型
        image_tensor: 预处理后的图像张量
        device: 计算设备
    
    Returns:
        dict: 包含预测类别和概率的字典
    """
    model.eval()
    
    # 添加batch维度
    image = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 前向传播
        outputs = model(image)
        
        # 计算概率（Softmax）
        probabilities = torch.softmax(outputs, dim=1)
        
        # 获取预测结果
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # 获取各类别的概率
        class_probs = probabilities[0].cpu().numpy()
    
    return {
        'predicted_class': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'all_probabilities': {
            CLASS_NAMES[i]: float(class_probs[i]) 
            for i in range(NUM_CLASSES)
        }
    }


def classify_image(image_path, model_path=None):
    """
    对单张图像进行分类的便捷函数
    
    Args:
        image_path: 图像文件路径
        model_path: 模型权重路径（可选，使用默认路径）
    
    Returns:
        dict: 预测结果
    """
    # 使用提供的模型路径或默认路径
    model_path = model_path or MODEL_PATH
    
    # 加载模型
    model = load_model(model_path, DEVICE)
    
    # 预处理图像
    transform = get_inference_transforms()
    image_tensor = preprocess_image(image_path, transform)
    
    # 推理
    result = predict_single_image(model, image_tensor, DEVICE)
    
    return result


# ==================== 主函数（命令行使用） ====================
def main():
    """命令行主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='糖尿病视网膜病变分级分类'
    )
    parser.add_argument(
        'image_path', 
        type=str, 
        help='待分类的图像路径'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=MODEL_PATH,
        help=f'模型权重路径（默认: {MODEL_PATH}）'
    )
    parser.add_argument(
        '--show-probabilities', 
        action='store_true',
        help='显示所有类别的概率'
    )
    
    args = parser.parse_args()
    
    # 检查图像文件
    if not os.path.exists(args.image_path):
        print(f"❌ 错误: 图像文件不存在: {args.image_path}")
        return
    
    print("=" * 50)
    print("🖼️  糖尿病视网膜病变分级分类")
    print("=" * 50)
    print(f"📷 输入图像: {args.image_path}")
    print(f"🖥️  使用设备: {DEVICE}")
    print()
    
    try:
        # 分类
        result = classify_image(args.image_path, args.model)
        
        # 输出结果
        print("=" * 50)
        print("📊 分类结果")
        print("=" * 50)
        print(f"🎯 预测类别: {result['class_name']}")
        print(f"📈 置信度: {result['confidence'] * 100:.2f}%")
        print()
        
        if args.show_probabilities:
            print("📊 各类别概率:")
            print("-" * 30)
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for class_name, prob in sorted_probs:
                bar = '█' * int(prob * 30)
                print(f"  {class_name:8s}: {prob * 100:6.2f}% {bar}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")


if __name__ == '__main__':
    main()

