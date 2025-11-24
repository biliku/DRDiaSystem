# -*- coding: utf-8 -*-
"""
病灶分割应用脚本
使用训练好的模型对眼底图像进行病灶分割
"""

import os
import warnings

# 设置环境变量
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

# 禁用警告
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =================================================================================
# 配置参数
# =================================================================================
# 图片路径（请修改为您的图片路径）
IMAGE_PATH = r"F:\DRDiaSys\DRDiaSys\django\DRDiaSys\datasets\dataset\aptos2019\test_images\e4dcca36ceb4.png"
# 模型路径
MODEL_PATH = r"F:\DRDiaSys\DRDiaSys\unet_epoch30.pth"
# 输出路径（可选，设置为 None 则不保存）
OUTPUT_PATH = r"F:\DRDiaSys\DRDiaSys\segmentation_result.png"

IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5  # 背景 + 4种病灶

# 颜色映射（RGB格式）- 与训练代码保持一致
# 训练代码使用 BGR: MA=蓝色(0,0,255), HE=绿色(0,255,0), EX=红色(255,0,0), SE=黄色(255,255,0)
# 转换为 RGB: MA=蓝色(0,0,255), HE=绿色(0,255,0), EX=红色(255,0,0), SE=黄色(255,255,0)
COLOR_MAP = {
    0: (0, 0, 0),        # 背景 - 黑色
    1: (0, 0, 255),      # MA (微动脉瘤) - 蓝色 (与训练代码一致)
    2: (0, 255, 0),      # HE (出血) - 绿色
    3: (255, 0, 0),      # EX (硬性渗出物) - 红色 (与训练代码一致)
    4: (255, 255, 0)     # SE (软性渗出物) - 黄色
}

# 病灶名称
LESION_NAMES = {
    0: '背景',
    1: 'MA (微动脉瘤)',
    2: 'HE (出血)',
    3: 'EX (硬性渗出物)',
    4: 'SE (软性渗出物)'
}

# =================================================================================
# 模型加载
# =================================================================================
def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = smp.Unet(
        "resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    )
    
    state_dict = torch.load(model_path, map_location=device)
    
    # 诊断信息：检查state_dict中的键
    print(f"模型文件包含 {len(state_dict)} 个参数")
    state_dict_keys = list(state_dict.keys())
    print(f"前10个键: {state_dict_keys[:10]}")
    
    # 检查是否缺少编码器权重
    has_encoder = any('encoder' in k for k in state_dict_keys)
    if not has_encoder:
        print("⚠️  警告: 模型文件中似乎缺少编码器权重")
        print("尝试使用 ImageNet 预训练权重初始化编码器...")
    
    # 尝试加载，允许部分匹配
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  缺少的键 ({len(missing_keys)} 个): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"⚠️  意外的键 ({len(unexpected_keys)} 个): {unexpected_keys[:5]}...")
        
        # 如果缺少编码器权重，使用预训练权重
        if missing_keys and any('encoder' in k for k in missing_keys):
            print("使用 ImageNet 预训练权重初始化缺失的编码器部分...")
            # 重新创建模型（会自动加载预训练权重）
            model = smp.Unet(
                "resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=NUM_CLASSES
            )
            # 再次加载保存的权重（这次应该能加载decoder部分）
            model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise
    
    model.to(device)
    model.eval()
    print(f"✓ 模型加载成功，使用设备: {device}")
    return model

# =================================================================================
# 图像处理和预测
# =================================================================================
def preprocess_image(image_path):
    """预处理图像 - 使用与训练时完全相同的预处理流程"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用与训练时完全相同的 transform（测试集模式，无数据增强）
    # 这确保了预测时的预处理与训练时完全一致
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # 应用 transform
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image']
    
    # 添加 batch 维度
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, image_rgb

def predict_mask(model, image_tensor, device):
    """使用模型进行预测"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        pred_mask = torch.argmax(outputs, dim=1)
        # 手动转换为 numpy，避免使用 .numpy() 方法
        # 先移到 CPU，然后转换为列表，再转换为 numpy 数组
        pred_mask_cpu = pred_mask[0].cpu()
        # 使用 tolist() 转换为 Python 列表，然后转换为 numpy 数组
        pred_mask_list = pred_mask_cpu.tolist()
        pred_mask_np = np.array(pred_mask_list, dtype=np.uint8)
    return pred_mask_np

# =================================================================================
# 可视化
# =================================================================================
def create_colored_mask(mask_np):
    """将类别掩码转换为彩色图像"""
    h, w = mask_np.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in COLOR_MAP.items():
        colored_mask[mask_np == cls_idx] = color
    return colored_mask

def create_overlay(original_image, mask_np, alpha=0.5):
    """创建原图与掩码的叠加图像"""
    if original_image.shape[:2] != mask_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (original_image.shape[1], original_image.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
    
    overlay = original_image.copy()
    for cls_idx in range(1, NUM_CLASSES):
        mask_cls = (mask_np == cls_idx).astype(np.uint8)
        if mask_cls.sum() > 0:
            color = COLOR_MAP[cls_idx]
            overlay[mask_cls > 0] = (alpha * np.array(color) + (1 - alpha) * overlay[mask_cls > 0]).astype(np.uint8)
    
    return overlay

def visualize_results(original_image, pred_mask_np, save_path=None):
    """可视化分割结果"""
    if original_image.shape[:2] != pred_mask_np.shape[:2]:
        pred_mask_np = cv2.resize(pred_mask_np, (original_image.shape[1], original_image.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    colored_mask = create_colored_mask(pred_mask_np)
    overlay = create_overlay(original_image, pred_mask_np)
    result_img = np.hstack([original_image, colored_mask, overlay])
    
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        print(f"✓ 结果已保存: {save_path}")
    
    return result_img

# =================================================================================
# 统计信息
# =================================================================================
def print_statistics(mask_np, image_name):
    """打印分割统计信息"""
    print(f"\n--- {os.path.basename(image_name)} 分割统计 ---")
    total_pixels = mask_np.size
    for cls_idx in range(NUM_CLASSES):
        count = np.sum(mask_np == cls_idx)
        percentage = (count / total_pixels) * 100
        print(f"  {LESION_NAMES[cls_idx]:<20}: {count:>8} 像素 ({percentage:>6.2f}%)")
    print()

# =================================================================================
# 主函数
# =================================================================================
def main():
    """主函数"""
    print("=" * 60)
    print("眼底图像病灶分割")
    print("=" * 60)
    
    # 检查图片路径
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 错误: 图片文件不存在: {IMAGE_PATH}")
        return
    
    # 加载模型
    model = load_model(MODEL_PATH, DEVICE)
    
    # 处理图像
    print(f"\n处理图像: {IMAGE_PATH}")
    image_tensor, original_image = preprocess_image(IMAGE_PATH)
    pred_mask_np = predict_mask(model, image_tensor, DEVICE)
    
    # 打印统计信息
    print_statistics(pred_mask_np, IMAGE_PATH)
    
    # 可视化并保存
    visualize_results(original_image, pred_mask_np, OUTPUT_PATH)
    
    print("✓ 处理完成！")

if __name__ == '__main__':
    main()
