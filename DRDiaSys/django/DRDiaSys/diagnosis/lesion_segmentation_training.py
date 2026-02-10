# -*- coding: utf-8 -*-
# 禁用 albumentations 的版本检查


import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from tqdm import tqdm
import warnings
from collections import OrderedDict # <<< 关键修改: 引入有序字典

os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

# =================================================================================
# 1. 配置参数
# =================================================================================
# <<< 关键修改: 使用r''原始字符串或'/'来处理Windows路径
IDRID_ROOT = r'F:\DRDiaSys\django\DRDiaSys\datasets\dataset\IDRiD'

# <<< 关键修改: 增加训练轮数
IMG_SIZE = 512
BATCH_SIZE = 6
EPOCHS = 65 # 增加到100轮
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# 修复损失函数 - 使用类别权重
class CombinedLoss(torch.nn.Module):
    def __init__(self, class_weights, dice_weight=0.7, ce_weight=0.3):  # 改进：提升Dice权重
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)  # 修复：使用权重
        self.focal_loss = FocalLoss(alpha=1, gamma=2)  # gamma=2专门处理难样本
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target)
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss

# <<< 关键修改: 使用OrderedDict确保病灶绘制顺序一致，防止小病灶被大病灶覆盖
LESION_MAP = OrderedDict([
    ('SE', ('4. Soft Exudates', 4)),  # 软性渗出物 (最稀有，优先)
    ('MA', ('1. Microaneurysms', 1)),  # 微动脉瘤
    ('HE', ('2. Haemorrhages', 2)),    # 出血
    ('EX', ('3. Hard Exudates', 3))    # 硬性渗出物
])
NUM_CLASSES = len(LESION_MAP) + 1

MODEL_SAVE_PATH = "django\DRDiaSys\diagnosis\\best_lesion_segmentation_model_v5.pth"
VISUAL_RESULT_DIR = "test_results_all" # 修改文件夹名以示区别


# =================================================================================
# 2. 数据增强与预处理 (无变化)
# =================================================================================
def get_transforms(is_train=True):
    if is_train:
        return A.Compose([ A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR), A.CLAHE(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7), A.RandomBrightnessContrast(p=0.5), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(), ])
    else:
        return A.Compose([ A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(), ])


# =================================================================================
# 3. PyTorch 数据集类 (修改后)
# =================================================================================
#训练与测试数据集
class IDRiDMultiClassDataset(Dataset):
    def __init__(self, image_names, root_dir, lesion_map, set_type, transform=None):
        self.image_names = image_names
        self.root_dir = root_dir
        self.lesion_map = lesion_map
        self.transform = transform
        set_folder_name = 'a. Training Set' if set_type == 'train' else 'b. Testing Set'
        self.images_base_path = os.path.join(self.root_dir, '1. Original Images', set_folder_name)
        self.groundtruths_base_path = os.path.join(self.root_dir, '2. All Segmentation Groundtruths', set_folder_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_base_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        base_name = os.path.splitext(img_name)[0]
        height, width, _ = image.shape
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # 使用OrderedDict确保绘制顺序
        for lesion_type, (folder_name, pixel_value) in self.lesion_map.items():
            mask_folder = os.path.join(self.groundtruths_base_path, folder_name)
            mask_file_name = f"{base_name}_{lesion_type}.png"
            mask_path = os.path.join(mask_folder, mask_file_name)
            if os.path.exists(mask_path):
                lesion_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 使用灰度模式读取
                if lesion_mask is not None:
                    # 确保掩码是二值的
                    lesion_mask = (lesion_mask > 0).astype(np.uint8)
                    combined_mask[lesion_mask > 0] = pixel_value
        
        if self.transform:
            augmented = self.transform(image=image, mask=combined_mask)
            image, mask = augmented['image'], augmented['mask']
            
        # <<< 关键修改: 返回图像名，方便可视化调试
        return image, mask.long(), img_name

# =================================================================================
# 4. 测试函数和可视化函数 (修复IoU计算)
# =================================================================================
def test_model(model, test_loader, device, model_path):
    print("\n" + "="*25); print("--- 开始在测试集上评估最终模型 ---")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 修复：正确初始化累计变量
    total_iou = 0.0
    total_dice = 0.0
    per_class_iou = torch.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="测试中"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            
            tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks, mode='multiclass', num_classes=NUM_CLASSES)
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            
            # 修复：正确累加
            total_iou += iou.cpu().item()
            total_dice += dice.cpu().item()
            
            # 每个类别的IoU
            iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
            per_class_iou += iou_per_class.mean(dim=0).cpu()

    # 修复：正确计算平均值
    avg_iou = total_iou / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    avg_per_class_iou = per_class_iou / len(test_loader)

    print("\n--- 测试结果报告 ---")
    print(f"  [*] 总体平均 IoU (交并比): {avg_iou:.4f}")
    print(f"  [*] 总体平均 Dice Score: {avg_dice:.4f}")
    print("\n  [*] 各类别平均 IoU:")
    lesion_names = ['背景'] + [k for k,v in LESION_MAP.items()]
    print(f"      - {lesion_names[0]:<4}: {avg_per_class_iou[0]:.4f}")
    for k,v in LESION_MAP.items():
        class_idx = v[1]
        print(f"      - {k:<4}: {avg_per_class_iou[class_idx]:.4f}")
    print("="*25)

def save_visual_results(model, loader, device, model_path):
    if not os.path.exists(VISUAL_RESULT_DIR): os.makedirs(VISUAL_RESULT_DIR)
    print(f"\n--- 保存所有测试样本的可视化结果至 '{VISUAL_RESULT_DIR}' 文件夹 ---")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # 颜色映射表：每个类别对应的BGR颜色值
    color_map = {
        0: (0,0,0),      # 背景 - 黑色
        1: (0,0,255),    # MA (Microaneurysms) - 微动脉瘤 - 蓝色
        2: (0,255,0),    # HE (Haemorrhages) - 出血 - 绿色
        3: (255,0,0),    # EX (Hard Exudates) - 硬性渗出物 - 红色
        4: (255,255,0)   # SE (Soft Exudates) - 软性渗出物 - 黄色
    }

    with torch.no_grad():
        for images, masks, img_names in tqdm(loader, desc="保存可视化结果"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)

            for j in range(images.size(0)):
                img_tensor = images[j]
                mask_np = masks[j].cpu().numpy()
                pred_mask_np = pred_masks[j].cpu().numpy()
                
                mean = torch.tensor([0.485, 0.456, 0.406], device=device)
                std = torch.tensor([0.229, 0.224, 0.225], device=device)
                
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = ((img_np * std.cpu().numpy() + mean.cpu().numpy()) * 255).astype(np.uint8)
                
                gt_color = np.zeros_like(img_np)
                pred_color = np.zeros_like(img_np)
                
                for cls_idx, color in color_map.items():
                    gt_color[mask_np == cls_idx] = color
                    pred_color[pred_mask_np == cls_idx] = color
                
                comparison_img = cv2.hconcat([img_np, gt_color, pred_color])
                save_path = os.path.join(VISUAL_RESULT_DIR, f"result_{os.path.splitext(img_names[j])[0]}.png")
                cv2.imwrite(save_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
    print(f"可视化结果保存完毕。")

# =================================================================================
# 5. 主执行流程 (修复模型一致性和损失函数)
# =================================================================================
def main():
    print(f"使用的设备: {DEVICE}")

    train_val_img_dir = os.path.join(IDRID_ROOT, '1. Original Images', 'a. Training Set')
    all_train_val_names = sorted([f for f in os.listdir(train_val_img_dir) if f.lower().endswith('.jpg')])
    train_names, val_names = train_test_split(all_train_val_names, test_size=0.2, random_state=42)
    test_img_dir = os.path.join(IDRID_ROOT, '1. Original Images', 'b. Testing Set')
    test_names = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith('.jpg')])
    
    train_dataset = IDRiDMultiClassDataset(train_names, IDRID_ROOT, LESION_MAP, set_type='train', transform=get_transforms(is_train=True))
    val_dataset = IDRiDMultiClassDataset(val_names, IDRID_ROOT, LESION_MAP, set_type='train', transform=get_transforms(is_train=False))
    test_dataset = IDRiDMultiClassDataset(test_names, IDRID_ROOT, LESION_MAP, set_type='test', transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"数据加载完成. 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    
    class_weights = torch.tensor([0.1, 13, 0.9, 1.9, 2.2], device=DEVICE)
    #                            背景  MA   HE   EX   SE
    # MA和SE权重提升到3.0+，因为它们最难检测

    print(f"使用的类别权重: {class_weights}")
    
    loss_fn = CombinedLoss(class_weights, dice_weight=0.7, ce_weight=0.3)  # 修复：传入权重

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # <<< 关键修改: 增加耐心值
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    best_iou = 0.0
    for epoch in range(EPOCHS):
        print("-" * 25); print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"训练", leave=False)
        for images, masks, _ in pbar_train:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(); outputs = model(images); loss = loss_fn(outputs, masks); loss.backward(); optimizer.step()
            train_loss += loss.item(); pbar_train.set_postfix(loss=loss.item())
        avg_train_loss = train_loss / len(train_loader)

        model.eval(); val_iou = 0
        pbar_val = tqdm(val_loader, desc=f"验证", leave=False)
        with torch.no_grad():
            for images, masks, _ in pbar_val:
                images, masks = images.to(DEVICE), masks.to(DEVICE); outputs = model(images)
                pred_masks = torch.argmax(outputs, dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks, mode='multiclass', num_classes=NUM_CLASSES)
                iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                val_iou += iou.item(); pbar_val.set_postfix(iou=iou.item())
    
        avg_val_iou = val_iou / len(val_loader)
        print(f"  平均训练损失: {avg_train_loss:.4f} | 平均验证IoU: {avg_val_iou:.4f}")
        lr_scheduler.step(avg_val_iou)
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> 新的最佳模型已保存，IoU: {best_iou:.4f}")

    print("\n" + "="*25); print("训练完成！")
    
    print("\n===加载测试数据===")
    test_img_dir = os.path.join(IDRID_ROOT, '1. Original Images', 'b. Testing Set')
    print(f"测试图像目录:{test_img_dir}")

    #检查目录是否存在
    if not os.path.exists(test_img_dir):
        print(f"错误:测试图像目录不存在:{test_img_dir}")
        exit(1)

    #获取所有图像文件
    all_files = os.listdir(test_img_dir)
    print(f"目录中的总文件数:{len(all_files)}")

    #筛选jpg文件
    test_names = sorted([f for f in all_files if f.lower().endswith('.jpg')])
    print(f"找到的jpg文件数:{len(test_names)}")

    if len(test_names) == 0:
        print("错误:没有找到任何jpg文件")
        exit(1)

    print("\n前5个测试图像文件名:")
    for name in test_names[:5]:
        print(f"-{name}")

    #创建测试数据集
    test_dataset = IDRiDMultiClassDataset(test_names, IDRID_ROOT, LESION_MAP, set_type='test', transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n测试数据集大小:{len(test_dataset)}")
    print(f"测试数据加载器批次数:{len(test_loader)}")

    #修复：保持模型配置一致
    test_model_instance = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES)
    test_model_instance = test_model_instance.to(DEVICE)  # 确保模型在正确的设备上
    test_model(test_model_instance, test_loader, DEVICE, MODEL_SAVE_PATH)
    save_visual_results(test_model_instance, test_loader, DEVICE, MODEL_SAVE_PATH)


if __name__ == '__main__':
    # <<< 关键修改: 修正启动方式，直接调用main函数
    # 禁用所有警告
    warnings.filterwarnings("ignore")
    os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

    main()
 

