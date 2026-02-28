# -*- coding: utf-8 -*-
"""
诊断服务模块 - 处理病灶分割等AI诊断任务
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import warnings
from collections import OrderedDict
from django.conf import settings
from django.utils import timezone
from datetime import datetime
import json
from .ai_models import AIModel

# 配置参数
IMG_SIZE = 512
BATCH_SIZE = 1  # 单张图像处理
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 病灶映射
LESION_MAP = OrderedDict([
    ('SE', ('4. Soft Exudates', 4)),
    ('MA', ('1. Microaneurysms', 1)),
    ('HE', ('2. Haemorrhages', 2)),
    ('EX', ('3. Hard Exudates', 3))
])
NUM_CLASSES = len(LESION_MAP) + 1

# 默认模型路径
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'best_lesion_segmentation_model_v4.pth'
)


class SingleImageDataset(Dataset):
    """单张图像数据集类"""
    
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # 读取图像
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {self.image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # 保存原始尺寸 (H, W)
        
        # 应用预处理
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, os.path.basename(self.image_path), original_size


def _single_image_collate_fn(batch):
    """自定义 collate，避免 DataLoader 将字符串/tuple 错误处理"""
    images, img_names, original_sizes = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(img_names), list(original_sizes)


def get_inference_transforms():
    """推理时的数据预处理"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(model_path=None):
    """加载训练好的模型（病灶分割）"""
    # 如果没有指定模型路径，则从数据库获取激活的模型
    if model_path is None:
        try:
            # 查询激活的生产环境模型
            active_model = AIModel.objects.filter(
                model_type='segmentation',
                is_active=True,
                status='production'
            ).first()

            if active_model and active_model.model_path and os.path.exists(active_model.model_path):
                model_path = active_model.model_path
                print(f"✓ 使用激活的病灶分割模型: {active_model.name} ({model_path})")
            else:
                # 没有激活模型或路径无效，使用默认路径
                model_path = DEFAULT_MODEL_PATH
                if active_model:
                    print(f"⚠ 激活的模型路径无效，使用默认模型: {model_path}")
                else:
                    print(f"⚠ 没有激活的病灶分割模型，使用默认模型: {model_path}")
        except Exception as e:
            # 数据库查询失败时使用默认路径
            print(f"⚠ 查询激活模型失败: {e}，使用默认模型")
            model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型
    model = smp.Unet("resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=NUM_CLASSES)

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def perform_lesion_segmentation(image_path, model_path=None, output_dir=None):
    """
    对单张图像进行病灶分割
    
    Args:
        image_path: 输入图像路径
        model_path: 模型路径（可选）
        output_dir: 输出目录（可选）
    
    Returns:
        dict: 包含分割结果和统计信息的字典
    """
    warnings.filterwarnings("ignore")
    
    # 检查输入文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(image_path)
    
    # 加载模型
    model = load_model(model_path)
    
    # 创建数据集和数据加载器
    transform = get_inference_transforms()
    dataset = SingleImageDataset(image_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=_single_image_collate_fn
    )
    
    # 颜色映射
    color_map = {
        0: (0, 0, 0),      # 背景 - 黑色
        1: (0, 0, 255),    # MA (Microaneurysms) - 微动脉瘤 - 蓝色
        2: (0, 255, 0),    # HE (Haemorrhages) - 出血 - 绿色
        3: (255, 0, 0),    # EX (Hard Exudates) - 硬性渗出物 - 红色
        4: (255, 255, 0)   # SE (Soft Exudates) - 软性渗出物 - 黄色
    }
    
    lesion_names = {
        0: "背景",
        1: "微动脉瘤(MA)",
        2: "出血(HE)",
        3: "硬渗出(EX)",
        4: "软渗出(SE)"
    }
    
    # 进行预测
    with torch.no_grad():
        for images, img_name, original_sizes in dataloader:
            images = images.to(DEVICE)
            
            # 模型推理
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            
            # 处理图像
            img_name = img_name[0]
            pred_mask = pred_masks[0].cpu().numpy()
            original_height, original_width = original_sizes[0]  # (H, W)
            original_height = int(original_height)
            original_width = int(original_width)
            
            # 反标准化原始图像
            img_tensor = images[0].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = ((img_np * std.numpy() + mean.numpy()) * 255).astype(np.uint8)
            
            # 调整回原始尺寸
            img_np = cv2.resize(img_np, (original_width, original_height))
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                 (original_width, original_height), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # 创建彩色分割结果
            colored_mask = np.zeros_like(img_np)
            for class_id, color in color_map.items():
                colored_mask[pred_mask == class_id] = color
            
            # 创建叠加图像
            overlay = cv2.addWeighted(img_np, 0.7, colored_mask, 0.3, 0)
            
            # 拼接结果图像：原图 | 分割结果 | 叠加图像
            result_img = np.hstack([img_np, colored_mask, overlay])
            
            # 保存结果图像
            base_name = os.path.splitext(img_name)[0]
            result_image_path = os.path.join(output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(result_image_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            
            # 保存分割掩码
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, pred_mask)
            
            # 统计各类别像素数量
            unique_classes, counts = np.unique(pred_mask, return_counts=True)
            total_pixels = pred_mask.size
            
            lesion_stats = {}
            for class_id, count in zip(unique_classes, counts):
                percentage = (count / total_pixels) * 100
                lesion_stats[int(class_id)] = {
                    'name': lesion_names[class_id],
                    'pixel_count': int(count),
                    'percentage': float(percentage)
                }
            
            return {
                'success': True,
                'result_image_path': result_image_path,
                'mask_path': mask_path,
                'lesion_statistics': lesion_stats,
                'image_name': img_name,
                'original_size': (int(original_height), int(original_width))
            }
    
    return {'success': False, 'error': '处理失败'}


# ==================== DR分级模型相关 ====================
import torchvision.models as models

# DR分级配置参数
DR_GRADING_IMG_SIZE = 384
DR_GRADING_NUM_CLASSES = 5
DR_GRADING_CLASS_NAMES = ['无病变', '轻度', '中度', '重度', '增殖性']

# 默认DR分级模型路径
DEFAULT_DR_GRADING_MODEL_PATH = r"F:\DRDiaSys\best_resnet_aptos_enhanced.pth"


class ImprovedResNetDR(nn.Module):
    """糖尿病视网膜病变分级模型"""
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


def get_dr_grading_transforms():
    """DR分级推理时的数据预处理"""
    return A.Compose([
        A.Resize(DR_GRADING_IMG_SIZE, DR_GRADING_IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_dr_grading_model(model_path=None):
    """加载DR分级模型"""
    # 如果没有指定模型路径，则从数据库获取激活的模型
    if model_path is None:
        try:
            # 查询激活的生产环境模型
            active_model = AIModel.objects.filter(
                model_type='grading',
                is_active=True,
                status='production'
            ).first()

            if active_model and active_model.model_path and os.path.exists(active_model.model_path):
                model_path = active_model.model_path
                print(f"✓ 使用激活的DR分级模型: {active_model.name} ({model_path})")
            else:
                # 没有激活模型或路径无效，使用默认路径
                model_path = DEFAULT_DR_GRADING_MODEL_PATH
                if active_model:
                    print(f"⚠ 激活的模型路径无效，使用默认模型: {model_path}")
                else:
                    print(f"⚠ 没有激活的DR分级模型，使用默认模型: {model_path}")
        except Exception as e:
            # 数据库查询失败时使用默认路径
            print(f"⚠ 查询激活模型失败: {e}，使用默认模型")
            model_path = DEFAULT_DR_GRADING_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DR分级模型文件不存在: {model_path}")

    # 创建模型
    model = ImprovedResNetDR(num_classes=DR_GRADING_NUM_CLASSES, pretrained=False)

    # 加载权重
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def perform_dr_grading(image_path, model_path=None, output_dir=None):
    """
    对单张图像进行DR分级
    
    Args:
        image_path: 输入图像路径
        model_path: 模型路径（可选）
        output_dir: 输出目录（可选）
    
    Returns:
        dict: 包含分级结果和概率的字典
    """
    warnings.filterwarnings("ignore")
    
    # 检查输入文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    # 加载模型
    model = load_dr_grading_model(model_path)
    
    # 预处理图像
    transform = get_dr_grading_transforms()
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 应用预处理
    transformed = transform(image=image)
    image_tensor = transformed['image']
    
    # 添加batch维度
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # 计算概率（Softmax）
        probabilities = torch.softmax(outputs, dim=1)
        
        # 获取预测结果
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # 获取各类别的概率
        class_probs = probabilities[0].cpu().numpy()
    
    # 构建结果
    result = {
        'success': True,
        'predicted_class': int(predicted_class),
        'class_name': DR_GRADING_CLASS_NAMES[predicted_class],
        'confidence': float(confidence),
        'all_probabilities': {
            DR_GRADING_CLASS_NAMES[i]: float(class_probs[i]) 
            for i in range(DR_GRADING_NUM_CLASSES)
        }
    }
    
    # 如果提供了输出目录，保存分级结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存分级结果为JSON
        import json
        result_json_path = os.path.join(output_dir, f"{base_name}_dr_grade.json")
        with open(result_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result['result_json_path'] = result_json_path
    
    return result
