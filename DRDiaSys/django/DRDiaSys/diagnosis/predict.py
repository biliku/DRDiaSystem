# -*- coding: utf-8 -*-
import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import warnings
from collections import OrderedDict

# =================================================================================
# 1. 配置参数 - 推理配置
# =================================================================================
# 模型配置（必须与训练时保持一致）
IMG_SIZE = 512
BATCH_SIZE = 4  # 推理时可以适当减小批次大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 病灶映射（必须与训练时保持一致）
LESION_MAP = OrderedDict([
    ('SE', ('4. Soft Exudates', 4)),
    ('MA', ('1. Microaneurysms', 1)),
    ('HE', ('2. Haemorrhages', 2)),
    ('EX', ('3. Hard Exudates', 3))
])
NUM_CLASSES = len(LESION_MAP) + 1

# 路径配置
MODEL_PATH = r"django\DRDiaSys\diagnosis\best_lesion_segmentation_model_v4.pth"
NEW_DATASET_DIR = r"F:\DRDiaSys\django\DRDiaSys\datasets\dataset\aptos2019_preprocessed\test_images_processed"  # 新数据集路径
RESULT_DIR = "prediction_results"  # 结果保存目录

# =================================================================================
# 2. 数据预处理 - 仅用于推理
# =================================================================================
def get_inference_transforms():
    """推理时的数据预处理（与训练时验证集保持一致）"""
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# =================================================================================
# 3. 推理专用数据集类
# =================================================================================
class InferenceDataset(Dataset):
    """专门用于推理的数据集类，只需要图像，不需要标注"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # 获取所有图像文件
        self.image_names = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_names.extend([f for f in os.listdir(image_dir) 
                                   if f.lower().endswith(ext.replace('*', ''))])
        
        self.image_names = sorted(self.image_names)
        print(f"找到 {len(self.image_names)} 张图像用于推理")
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # 保存原始尺寸 (H, W)
        
        # 应用预处理
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, img_name, original_size

def custom_collate_fn(batch):
    """自定义批处理函数"""
    images = []
    img_names = []
    original_sizes = []
    
    for item in batch:
        images.append(item[0])
        img_names.append(item[1])
        original_sizes.append(item[2])
    
    images = torch.stack(images, dim=0)
    return images, img_names, original_sizes

# =================================================================================
# 4. 推理和结果保存函数
# =================================================================================
def load_model(model_path, device):
    """加载训练好的模型"""
    print(f"🔄 加载模型: {model_path}")
    
    # 创建模型（必须与训练时保持一致）
    model = smp.Unet("resnet34", encoder_weights="imagenet", 
                     in_channels=3, classes=NUM_CLASSES)
    
    # 加载权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功，设备: {device}")
    return model

def predict_images(model, dataloader, device, result_dir):
    """对图像进行预测并保存结果"""
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 颜色映射
    color_map = {
        0: (0,0,0),      # 背景 - 黑色
        1: (0,0,255),    # MA (Microaneurysms) - 微动脉瘤 - 蓝色
        2: (0,255,0),    # HE (Haemorrhages) - 出血 - 绿色
        3: (255,0,0),    # EX (Hard Exudates) - 硬性渗出物 - 红色
        4: (255,255,0)   # SE (Soft Exudates) - 软性渗出物 - 黄色
    }
    
    lesion_names = {
        0: "背景",
        1: "微动脉瘤(MA)",
        2: "出血(HE)", 
        3: "硬渗出(EX)",
        4: "软渗出(SE)"
    }
    
    print(f"🎨 开始预测，结果将保存到: {result_dir}")
    
    with torch.no_grad():
        for batch_idx, (images, img_names, original_sizes) in enumerate(tqdm(dataloader, desc="预测中")):
            images = images.to(device)
            
            # 模型推理
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            
            # 处理每张图像
            for i in range(images.size(0)):
                img_name = img_names[i]
                pred_mask = pred_masks[i].cpu().numpy()
                original_size = original_sizes[i]  # (H, W)
                
                # 反标准化原始图像
                img_tensor = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = ((img_np * std.numpy() + mean.numpy()) * 255).astype(np.uint8)
                
                # 调整回原始尺寸 (W, H) - OpenCV格式
                original_width = original_size[1]
                original_height = original_size[0]
                
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
                
                # 保存结果
                base_name = os.path.splitext(img_name)[0]
                save_path = os.path.join(result_dir, f"{base_name}_prediction.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                
                # 统计各类别像素数量
                unique_classes, counts = np.unique(pred_mask, return_counts=True)
                total_pixels = pred_mask.size
                
                print(f"\n📊 {img_name} 预测统计:")
                for class_id, count in zip(unique_classes, counts):
                    percentage = (count / total_pixels) * 100
                    print(f"   {lesion_names[class_id]}: {count:,} 像素 ({percentage:.3f}%)")

def main():
    """主函数"""
    print("🚀 启动眼底图像病灶分割推理")
    print(f"使用设备: {DEVICE}")
    
    # 检查输入目录
    if not os.path.exists(NEW_DATASET_DIR):
        print(f"❌ 错误: 新数据集目录不存在: {NEW_DATASET_DIR}")
        return
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 模型文件不存在: {MODEL_PATH}")
        return
    
    # 创建数据集和数据加载器
    dataset = InferenceDataset(NEW_DATASET_DIR, transform=get_inference_transforms())
    if len(dataset) == 0:
        print("❌ 错误: 没有找到任何图像文件")
        return
    
    # 使用自定义的批处理函数
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    
    # 加载模型
    model = load_model(MODEL_PATH, DEVICE)
    
    # 进行预测
    predict_images(model, dataloader, DEVICE, RESULT_DIR)
    
    print(f"\n🎉 预测完成！")
    print(f"📁 结果保存在: {RESULT_DIR}")
    print(f"💡 图像格式: [原图 | 分割结果 | 叠加图像]")

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
