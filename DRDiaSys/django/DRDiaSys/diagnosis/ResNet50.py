import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tqdm import tqdm
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
warnings.filterwarnings('ignore')

# --- 全局常量和配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

IMG_SIZE = 384
BATCH_SIZE = 16  # 减少batch size
NUM_WORKERS = 4
NUM_EPOCHS = 15  # 大幅减少epoch
LEARNING_RATE = 1e-5  # 显著降低学习率
NUM_CLASSES = 5
MODEL_SAVE_PATH = "best_resnet_aptos_improved.pth"
DATA_ROOT = 'F:\\DRDiaSys\\django\\DRDiaSys\\datasets\\dataset\\aptos2019_preprocessed'
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, 'train.csv')
VAL_CSV_PATH = os.path.join(DATA_ROOT, 'valid.csv')
TEST_CSV_PATH = os.path.join(DATA_ROOT, 'test.csv')

TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, 'train_images_processed')
VAL_IMAGE_DIR = os.path.join(DATA_ROOT, 'val_images_processed')
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, 'test_images_processed')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- 最小化数据增强 ---
class AdvancedDataTransforms:
    def __init__(self):
        # 极简数据增强
        self.train_transforms = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# --- 改进的数据集类 ---
class ImprovedDRDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, use_albumentations=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.use_albumentations = use_albumentations
        
        # 计算类别权重
        self.class_counts = self.df['diagnosis'].value_counts().sort_index()
        print(f"类别分布: {self.class_counts.to_dict()}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['id_code']
        if not image_name.endswith('.png'):
            image_name += '.png'
        
        image_path = os.path.join(self.image_dir, image_name)
        
        if self.use_albumentations:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        label = int(self.df.iloc[idx]['diagnosis'])
        return image, label
    
    def get_class_weights(self):
        """计算类别权重用于损失函数"""
        labels = self.df['diagnosis'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        return torch.FloatTensor(class_weights)

# --- 极度简化的模型架构 ---
class ImprovedResNetDR(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.7):
        super(ImprovedResNetDR, self).__init__()
        
        # 使用ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)
        
        # 冻结前面的层
        for name, param in self.backbone.named_parameters():
            if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        
        # 简化分类头
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# --- 混合损失函数 ---
class MixedLoss(nn.Module):
    def __init__(self, alpha=0.7, class_weights=None):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        # 将分类问题视为回归问题
        pred_soft = F.softmax(pred, dim=1)
        target_reg = target.float().unsqueeze(1)
        pred_reg = torch.sum(pred_soft * torch.arange(5).float().to(pred.device), dim=1, keepdim=True)
        mse = self.mse_loss(pred_reg, target_reg)
        
        return self.alpha * ce + (1 - self.alpha) * mse

# --- 改进的训练函数 ---
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=3):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    early_stop_counter = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc='训练中')
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='验证中')
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # 过拟合检测
        overfitting_gap = train_acc - val_acc
        if overfitting_gap > 15:
            print(f'⚠️  过拟合警告：差距 {overfitting_gap:.2f}%')
        
        # 早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'✅ 保存最佳模型，验证准确率: {best_val_acc:.2f}%')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # 严重过拟合时提前终止
        if overfitting_gap > 30:
            print(f'🚫 严重过拟合，提前终止训练！')
            break
            
        if early_stop_counter >= patience:
            print(f'⏹️  早停触发！')
            break
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# --- TTA评估 ---
def evaluate_model_with_tta(model, test_loader, num_tta=3):
    model.eval()
    all_predictions = []
    all_labels = []
    
    # TTA变换
    tta_transforms = [
        A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=1.0 if i == 1 else 0.0),
            A.Rotate(limit=5 if i == 2 else 0, p=1.0 if i == 2 else 0.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]) for i in range(num_tta)
    ]
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='测试中(TTA)')
        for images, labels in test_bar:
            batch_predictions = []
            
            for img, label in zip(images, labels):
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                tta_outputs = []
                for transform in tta_transforms:
                    transformed = transform(image=img_np)
                    tta_img = transformed['image'].unsqueeze(0).to(DEVICE)
                    output = model(tta_img)
                    tta_outputs.append(F.softmax(output, dim=1))
                
                avg_output = torch.mean(torch.cat(tta_outputs, dim=0), dim=0)
                predicted = torch.argmax(avg_output).cpu().numpy()
                batch_predictions.append(predicted)
                all_labels.append(label.numpy())
            
            all_predictions.extend(batch_predictions)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    class_names = ['无病变', '轻度', '中度', '重度', '增殖性']
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    return accuracy, all_predictions, all_labels, report

# --- 主函数 ---
def main():
    # 创建数据变换
    transforms_obj = AdvancedDataTransforms()
    
    # 创建数据集
    print("正在加载数据集...")
    train_dataset = ImprovedDRDataset(TRAIN_CSV_PATH, TRAIN_IMAGE_DIR, transforms_obj.train_transforms)
    val_dataset = ImprovedDRDataset(VAL_CSV_PATH, VAL_IMAGE_DIR, transforms_obj.val_transforms)
    test_dataset = ImprovedDRDataset(TEST_CSV_PATH, TEST_IMAGE_DIR, transforms_obj.val_transforms)
    
    # 获取类别权重
    class_weights = train_dataset.get_class_weights().to(DEVICE)
    print(f"类别权重: {class_weights}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                             drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    print("正在初始化模型...")
    model = ImprovedResNetDR(num_classes=NUM_CLASSES, pretrained=True, dropout_rate=0.7)
    model = model.to(DEVICE)
    
    # 使用混合损失
    criterion = MixedLoss(alpha=0.7, class_weights=class_weights)
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, patience=3
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 测试
    print("正在加载最佳模型进行测试...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # 使用TTA评估模型
    test_accuracy, predictions, true_labels, report = evaluate_model_with_tta(model, test_loader, num_tta=3)
    
    print(f"\n🎯 测试准确率 (TTA): {test_accuracy:.4f}")
    print("\n分类报告:")
    class_names = ['无病变', '轻度', '中度', '重度', '增殖性']
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1_score = report[str(i)]['f1-score']
            print(f"{class_name}: 精确率={precision:.3f}, 召回率={recall:.3f}, F1分数={f1_score:.3f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions)
    
    print("改进训练完成！")

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='训练准确率', color='blue')
    ax2.plot(val_accuracies, label='验证准确率', color='red')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(true_labels, predictions):
    class_names = ['无病变', '轻度', '中度', '重度', '增殖性']
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('改进模型混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
