import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.multiprocessing
import logging
from datetime import datetime
from collections import Counter
import torch.nn.functional as F

# --- 全局常量和配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

IMG_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
MODEL_SAVE_PATH = "best_efficientnetb3_aptos.pth"
DATA_ROOT = 'F:\\DRDiaSys\\django\\DRDiaSys\\datasets\\dataset\\aptos2019_preprocessed'
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, 'train.csv')
VAL_CSV_PATH = os.path.join(DATA_ROOT, 'valid.csv')
TEST_CSV_PATH = os.path.join(DATA_ROOT, 'test.csv')

TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, 'train_images_processed')
VAL_IMAGE_DIR = os.path.join(DATA_ROOT, 'val_images_processed')
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, 'test_images_processed')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 日志设置 ---
def setup_logger():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_balanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = None

# --- 修改1: 数据增强策略 - 针对少数类的额外增强 ---
def create_balanced_dataset(df, multiplier_dict={0: 1, 1: 8, 2: 4, 3: 15, 4: 20}):
    """通过数据增强平衡数据集"""
    balanced_df = []
    
    for class_id in range(NUM_CLASSES):
        class_samples = df[df['diagnosis'] == class_id]
        multiplier = multiplier_dict.get(class_id, 1)
        
        # 重复样本
        for _ in range(multiplier):
            balanced_df.append(class_samples)
    
    result_df = pd.concat(balanced_df, ignore_index=True).sample(frac=1).reset_index(drop=True)
    logger.info(f"平衡后数据集大小: {len(result_df)}")
    logger.info(f"平衡后类别分布:\n{result_df['diagnosis'].value_counts().sort_index()}")
    
    return result_df

# --- 修改2: 极激进的类别权重计算 ---
def calculate_class_weights(df):
    """使用平方根倒数计算极激进的类别权重"""
    class_counts = df['diagnosis'].value_counts().sort_index()
    total_samples = len(df)
    weights = []
    
    logger.info("=== 极激进类别权重计算 ===")
    for i in range(NUM_CLASSES):
        if i in class_counts.index:
            count = class_counts[i]
            # 极激进：使用平方根倒数
            base_weight = total_samples / (NUM_CLASSES * count)
            aggressive_weight = np.sqrt(base_weight) * 2  # 平方根 + 额外放大
            weights.append(aggressive_weight)
            logger.info(f"类别 {i}: {count} 样本, 权重: {aggressive_weight:.4f}")
        else:
            weights.append(1.0)
            logger.info(f"类别 {i}: 0 样本, 权重: 1.0000")
    
    return torch.FloatTensor(weights)

# --- 修改3: 三阶段渐进损失函数 ---
class ProgressiveLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
        self.stage = 1  # 默认从阶段1开始

    def set_stage(self, stage):
        self.stage = stage

    def forward(self, outputs, targets):
        mse = F.mse_loss(outputs, targets, reduction='none')
        sample_weights = torch.ones_like(targets)
        for i, weight in enumerate(self.class_weights):
            sample_weights[targets == i] = weight

        if self.stage == 1:
            # 阶段1: 温和的加权MSE
            weighted_mse = mse.squeeze() * sample_weights
            return weighted_mse.mean()
        
        elif self.stage == 2:
            # 阶段2: 中等Focal Loss
            focal_weight = 1.5 * (mse.detach() + 1e-8) ** 1.0
            combined_weight = sample_weights * focal_weight.squeeze()
            weighted_mse = mse.squeeze() * combined_weight
            return weighted_mse.mean()
        
        else:  # stage == 3
            # 阶段3: 极激进Focal Loss
            focal_weight = 3.0 * (mse.detach() + 1e-8) ** 1.5
            combined_weight = sample_weights * focal_weight.squeeze()
            weighted_mse = mse.squeeze() * combined_weight
            return weighted_mse.mean()

# --- 修改4: 自适应阈值系统 ---
class AdaptiveThresholdSystem:
    def __init__(self):
        self.thresholds = [0.5, 1.5, 2.5, 3.5]  # 初始阈值
        self.correct_predictions = [0] * NUM_CLASSES
        self.total_samples = [1] * NUM_CLASSES  # 避免除零
        
    def update_stats(self, true_labels, pred_labels):
        """更新统计信息"""
        for true_label, pred_label in zip(true_labels, pred_labels):
            # 将 numpy.float32 转换为整数
            true_label = int(true_label)
            pred_label = int(pred_label)
            
            self.total_samples[true_label] += 1
            if true_label == pred_label:
                self.correct_predictions[true_label] += 1
    
    def adapt_thresholds(self):
        """根据各类别准确率动态调整阈值"""
        accuracy_rates = [correct/total for correct, total in zip(self.correct_predictions, self.total_samples)]
        
        # 对于准确率低的类别，调整阈值
        for i in range(len(self.thresholds)):
            if accuracy_rates[i] < 0.7:  # 如果准确率低于70%
                # 降低阈值，使模型更容易预测为该类别
                self.thresholds[i] *= 0.95
            elif accuracy_rates[i] > 0.9:  # 如果准确率高于90%
                # 提高阈值，使模型更谨慎
                self.thresholds[i] *= 1.05
                
        # 确保阈值在合理范围内
        self.thresholds = [max(0.1, min(3.9, t)) for t in self.thresholds]
        
        # 重置统计信息
        self.correct_predictions = [0] * NUM_CLASSES
        self.total_samples = [1] * NUM_CLASSES
    
    def get_thresholds(self):
        """获取当前阈值"""
        return self.thresholds

# 全局自适应阈值系统
adaptive_threshold_system = AdaptiveThresholdSystem()

# --- 修改5: 改进的输出转换函数 ---
def convert_outputs_to_class(outputs, use_adaptive=True):
    """使用自适应阈值进行分类"""
    if use_adaptive:
        thresholds = adaptive_threshold_system.thresholds
    else:
        thresholds = [0.5, 1.5, 2.5, 3.5]
    
    outputs = torch.clamp(outputs.squeeze(dim=1), 0, 4)
    classes = torch.zeros_like(outputs, dtype=torch.long)
    
    for i, threshold in enumerate(thresholds):
        classes = torch.where(outputs > threshold, i+1, classes)
    
    return classes

# --- 数据集定义（保持不变）---
class AptosDataset(Dataset):
    def __init__(self, df, transform=None, image_dir=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'id_code'] + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        label = self.df.loc[idx, 'diagnosis']
        
        try:
            image = cv2.imread(img_path)
            if image is None: 
                raise RuntimeError(f"无法读取图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform: 
                image = self.transform(image=image)['image']
            
            return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        except Exception as e:
            if logger: 
                logger.error(f"加载图像 {img_path} 时出错: {e}")
            default_image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if self.transform: 
                default_image = self.transform(image=default_image)['image']
            return default_image, torch.tensor(0.0, dtype=torch.float32).unsqueeze(0)

# --- 数据增强定义（保持不变）---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.7),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
        A.RandomGamma(gamma_limit=(80, 120), p=1),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1)
    ], p=0.5),
    A.OneOf([
        A.GaussNoise(p=1), 
        A.GaussianBlur(blur_limit=(3, 7), p=1), 
        A.MotionBlur(blur_limit=(3, 7), p=1)
    ], p=0.3),
    A.OneOf([
        A.OpticalDistortion(distort_limit=0.2, p=1), 
        A.ElasticTransform(alpha=1, sigma=50, p=1), 
        A.CoarseDropout(p=0.5)
    ], p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

val_test_transform = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# --- 详细评估函数（保持不变）---
def detailed_class_evaluation(all_labels, all_preds):
    """详细的每类别评估"""
    logger.info("\n=== 详细类别评估 ===")
    
    for class_id in range(NUM_CLASSES):
        class_mask = np.array(all_labels) == class_id
        if class_mask.any():
            total_class_samples = class_mask.sum()
            class_preds = np.array(all_preds)[class_mask]
            correct_preds = np.sum(class_preds == class_id)
            class_accuracy = correct_preds / total_class_samples
            
            pred_distribution = {}
            for pred_class in range(NUM_CLASSES):
                count = np.sum(class_preds == pred_class)
                pred_distribution[pred_class] = count
            
            logger.info(f"类别 {class_id}: {correct_preds}/{total_class_samples} = {class_accuracy:.4f}")
            logger.info(f"  预测分布: {pred_distribution}")
        else:
            logger.info(f"类别 {class_id}: 无测试样本")
    
    report = classification_report(all_labels, all_preds, target_names=[f'Class_{i}' for i in range(NUM_CLASSES)])
    logger.info(f"\n分类报告:\n{report}")

# --- 修改6: 三阶段训练函数 ---
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [训练]", leave=False, position=0)
    
    # 动态调整损失函数阶段
    if hasattr(criterion, 'set_stage'):
        if epoch < total_epochs * 0.3:
            criterion.set_stage(1)
        elif epoch < total_epochs * 0.7:
            criterion.set_stage(2)
        else:
            criterion.set_stage(3)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        
        preds = convert_outputs_to_class(outputs, use_adaptive=True)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.squeeze(dim=1).cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # 更新自适应阈值系统
    adaptive_threshold_system.update_stats(all_labels, all_preds)
    if epoch % 5 == 0:  # 每5轮调整一次阈值
        adaptive_threshold_system.adapt_thresholds()
        
    if not all_labels: 
        return 0.0, 0.0
    
    avg_loss = running_loss / len(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, kappa

def validate_one_epoch(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [验证]", leave=False, position=0)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds = convert_outputs_to_class(outputs, use_adaptive=True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze(dim=1).cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    if not all_labels: 
        return 0.0, 0.0
    
    avg_loss = running_loss / len(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, kappa

def evaluate_model(model, dataloader, device, model_path=None):
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"已加载模型: {model_path}")
    
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="评估中", leave=False):
            images = images.to(device)
            outputs = model(images)
            
            preds = convert_outputs_to_class(outputs, use_adaptive=True)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.squeeze(dim=1).cpu().numpy())
    
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
    
    logger.info(f"\n=== 评估结果 ===")
    logger.info(f"整体准确率: {accuracy:.4f}")
    logger.info(f"Quadratic Kappa: {kappa:.4f}")
    logger.info(f"\n混淆矩阵:\n{conf_mat}")
    
    detailed_class_evaluation(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig('confusion_matrix_balanced_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    global logger
    logger = setup_logger()
    
    if not torch.cuda.is_available(): 
        logger.error("未检测到GPU!")
        return
    
    logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")

    train_df_orig = pd.read_csv(TRAIN_CSV_PATH)
    val_df = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    logger.info("\n=== 原始训练集类别分布 ===")
    logger.info(f"{train_df_orig['diagnosis'].value_counts().sort_index()}")
    
    # 修改7: 创建平衡数据集
    balanced_train_df = create_balanced_dataset(train_df_orig)
    
    class_weights = calculate_class_weights(train_df_orig)  # 仍基于原始分布计算权重
    
    train_dataset = AptosDataset(df=balanced_train_df, transform=train_transform, image_dir=TRAIN_IMAGE_DIR)
    val_dataset = AptosDataset(df=val_df, transform=val_test_transform, image_dir=VAL_IMAGE_DIR)
    test_dataset = AptosDataset(df=test_df, transform=val_test_transform, image_dir=TEST_IMAGE_DIR)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,  # 使用shuffle而不是weighted sampler
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    logger.info(f"\nDataLoaders创建完成，使用平衡数据集 + 渐进损失 + 自适应阈值")

    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1).to(DEVICE)
    
    # 修改8: 使用渐进损失函数
    criterion = ProgressiveLoss(class_weights.to(DEVICE))
    
    # 修改9: 调整优化器参数
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)
    
    logger.info(f"\n开始三阶段渐进训练:")
    logger.info(f"阶段1 (0-30%): 温和加权MSE")
    logger.info(f"阶段2 (30-70%): 中等Focal Loss") 
    logger.info(f"阶段3 (70-100%): 极激进Focal Loss")
    logger.info(f"自适应阈值系统: 每5轮根据类别错误率调整阈值")
    
    best_val_kappa = -1.0
    patience, no_improve_epochs = 8, 0
    history = {'train_loss': [], 'val_loss': [], 'train_kappa': [], 'val_kappa': []}

    for epoch in range(NUM_EPOCHS):
        train_loss, train_kappa = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, epoch, NUM_EPOCHS)
        val_loss, val_kappa = validate_one_epoch(model, val_loader, criterion, DEVICE, epoch)
        
        scheduler.step(val_kappa)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_kappa'].append(train_kappa) 
        history['val_kappa'].append(val_kappa)
        
        tqdm.write(f"\nEpoch {epoch+1}/{NUM_EPOCHS} 结果:")
        tqdm.write(f"  [训练] -> 损失: {train_loss:.4f}, Kappa: {train_kappa:.4f}")
        tqdm.write(f"  [验证] -> 损失: {val_loss:.4f}, Kappa: {val_kappa:.4f}")
        tqdm.write(f"  当前学习率: {optimizer.param_groups[0]['lr']:.2e}")
        tqdm.write(f"  当前阈值: {adaptive_threshold_system.thresholds}")
        
        if val_kappa > best_val_kappa + 1e-6:
            best_val_kappa = val_kappa
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            tqdm.write(f"  🎉 保存最佳模型 - 验证集 Kappa: {best_val_kappa:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            tqdm.write(f"  验证集 Kappa 未提升 ({no_improve_epochs}/{patience})")
            
            if no_improve_epochs >= patience:
                logger.info(f"\n{patience} 轮未改善，提前停止训练")
                break

    logger.info("\n训练完成!")
    
    epochs_ran = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, history['train_loss'], label='训练损失')
    plt.plot(epochs_ran, history['val_loss'], label='验证损失')
    plt.title('损失曲线 (渐进损失)')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, history['train_kappa'], label='训练Kappa')
    plt.plot(epochs_ran, history['val_kappa'], label='验证Kappa')
    plt.title('Kappa分数曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Kappa分数')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("\n=== 在测试集上评估最佳模型 ===")
    evaluate_model(model, test_loader, DEVICE, MODEL_SAVE_PATH)
    
    logger.info(f"\n=== 训练和评估完成 ===")
    logger.info(f"最佳模型已保存至: {MODEL_SAVE_PATH}")
    logger.info(f"训练曲线已保存至: training_curves.png")
    logger.info(f"混淆矩阵已保存至: confusion_matrix.png")

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: 
        pass
    main()
