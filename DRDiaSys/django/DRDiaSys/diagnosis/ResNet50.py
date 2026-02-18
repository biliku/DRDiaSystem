import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

# --- 超参数配置（增强版） ---
IMG_SIZE = 384
BATCH_SIZE = 32  # 增大batch size提升稳定性
NUM_WORKERS = 4
NUM_EPOCHS = 60  # 进一步延长训练时间
LEARNING_RATE = 1e-4  # 基础学习率，配合分层学习率策略
NUM_CLASSES = 5
MODEL_SAVE_PATH = "best_resnet_aptos_enhanced.pth"  # 使用新的模型保存路径
# use raw string to avoid escape sequences like \a being interpreted
DATA_ROOT = r'F:\DRDiaSys\DRDiaSys\django\DRDiaSys\datasets\dataset\aptos2019_preprocessed'
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, 'train.csv')
VAL_CSV_PATH = os.path.join(DATA_ROOT, 'valid.csv')
TEST_CSV_PATH = os.path.join(DATA_ROOT, 'test.csv')

TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, 'train_images_processed')
VAL_IMAGE_DIR = os.path.join(DATA_ROOT, 'val_images_processed')
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, 'test_images_processed')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- 眼底图像专用数据增强 ---
class AdvancedDataTransforms:
    """
    针对眼底图像特点设计的增强策略：
    1. 模拟不同拍摄条件（亮度、对比度、颜色）
    2. 模拟图像几何变换（旋转、翻转）
    3. 模拟病变区域变化
    """
    def __init__(self):
        # 训练集：丰富的增强策略
        self.train_transforms = A.Compose([
            # 几何变换
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),
            
            # 亮度/对比度/颜色增强（模拟不同相机和曝光条件）
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=1
                ),
            ], p=0.6),
            
            # 颜色变换（模拟不同设备）
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.3
            ),
            
            # 模糊增强（模拟低质量图像）
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1),
                A.GaussianBlur(blur_limit=(3, 5), p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.2),
            
            # 标准化（ImageNet标准）
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # 验证/测试集：仅标准化
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

# --- 改进的模型架构 ---
class ImprovedResNetDR(nn.Module):
    """
    改进的糖尿病视网膜病变分类模型：
    - 基于ResNet34骨干网络（ImageNet预训练）
    - 简化的分类头（避免过度正则化）
    - 分层微调策略
    """
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.5):
        super(ImprovedResNetDR, self).__init__()
        
        # 使用ResNet34作为骨干网络
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 获取特征维度
        num_features = self.backbone.fc.in_features
        
        # 简化的分类头（避免过度正则化）
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # 适度的Dropout
            nn.Linear(num_features, num_classes)
        )
        
        # 初始化新添加的分类头权重
        self._init_classifier()
    
    def _init_classifier(self):
        """初始化分类头权重"""
        for m in self.backbone.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_layer_groups(self):
        """获取层分组用于分层学习率"""
        # 分层策略：浅层低学习率，深层高学习率
        backbone = self.backbone
        
        # Layer1-2: 冻结，使用极低学习率
        layer1_params = []
        layer2_params = []
        
        # Layer3-4: 微调，使用中等学习率
        layer3_params = []
        layer4_params = []
        
        # FC层: 高学习率
        fc_params = []
        
        for name, param in backbone.named_parameters():
            if 'layer1' in name:
                layer1_params.append(param)
            elif 'layer2' in name:
                layer2_params.append(param)
            elif 'layer3' in name:
                layer3_params.append(param)
            elif 'layer4' in name:
                layer4_params.append(param)
            elif 'fc' in name:
                fc_params.append(param)
        
        return [
            {'params': layer1_params, 'lr': LEARNING_RATE * 0.1},
            {'params': layer2_params, 'lr': LEARNING_RATE * 0.5},
            {'params': layer3_params, 'lr': LEARNING_RATE * 1.0},
            {'params': layer4_params, 'lr': LEARNING_RATE * 2.0},
            {'params': fc_params, 'lr': LEARNING_RATE * 5.0},
        ]
    
    def forward(self, x):
        return self.backbone(x)

# --- 简化的损失函数（仅使用 CE + Label Smoothing） ---
class SimpleLoss(nn.Module):
    """
    简化的损失函数：
    - 仅使用 Label Smoothing CE Loss
    - 避免 KL Loss 的缩放问题
    """
    def __init__(self, class_weights=None, label_smoothing=0.1):
        super(SimpleLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, 
            label_smoothing=label_smoothing
        )
        
    def forward(self, pred, target):
        return self.ce_loss(pred, target)


# --- 简化的训练函数（无增强） ---
def train_model_simple(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """
    简化的训练函数：
    - 移除增强技术，专注于稳定训练
    - 每个epoch保存最佳模型
    """
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 7
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        
        train_bar = tqdm(train_loader, desc='训练中')
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
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
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='验证中')
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'当前学习率: {current_lr:.2e}')
        
        # 过拟合检测
        overfitting_gap = train_acc - val_acc
        if overfitting_gap > 10:
            print(f'⚠️  过拟合警告：差距 {overfitting_gap:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'✅ 保存最佳模型，验证准确率: {best_val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 学习率调度
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            if patience_counter >= patience:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.5, 1e-7)
                print(f'📉 降低学习率至: {optimizer.param_groups[0]["lr"]:.2e}')
                patience_counter = 0
        
        # 保存最终模型
        if current_lr <= 1e-7 and epoch > num_epochs - 5:
            torch.save(model.state_dict(), f"final_model_epoch_{epoch+1}.pth")
            print(f'💾 保存最终模型 (epoch {epoch+1})')
    
    print(f'\n🏆 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})')
    return train_losses, val_losses, train_accuracies, val_accuracies

# --- TTA评估（增强版） ---
def evaluate_model_with_tta(model, test_loader, num_tta=5):
    """
    增强版测试时数据评估：
    - 使用多种变换组合
    - 对预测结果进行软投票
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    # TTA变换组合（更多样化）
    tta_transforms = [
        # 原始
        A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 水平翻转
        A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 小角度旋转
        A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Rotate(limit=10, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 垂直翻转
        A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 亮度调整
        A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ][:num_tta]
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='测试中(TTA)')
        for images, labels in test_bar:
            batch_predictions = []
            
            for img, label in zip(images, labels):
                # 反标准化以便应用albumentations变换
                img_np = img.permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                tta_outputs = []
                for transform in tta_transforms:
                    transformed = transform(image=img_np)
                    tta_img = transformed['image'].unsqueeze(0).to(DEVICE)
                    output = model(tta_img)
                    tta_outputs.append(F.softmax(output, dim=1))
                
                # 软投票：平均概率后取最大
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
    model = ImprovedResNetDR(num_classes=NUM_CLASSES, pretrained=True, dropout_rate=0.5)
    model = model.to(DEVICE)
    
    # 使用简单损失函数（Label Smoothing）
    criterion = SimpleLoss(class_weights=class_weights, label_smoothing=0.1)
    
    # 使用分层学习率 + AdamW优化器
    optimizer = optim.AdamW(model.get_layer_groups(), lr=LEARNING_RATE, weight_decay=1e-2)
    
    # 学习率调度：ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 简化训练（无增强）
    print("=" * 60)
    print("开始稳定训练...")
    print("特性: Label Smoothing + 稳定训练流程")
    print("=" * 60)
    train_losses, val_losses, train_accuracies, val_accuracies = train_model_simple(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 测试
    print("正在加载最佳模型进行测试...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # 使用TTA评估模型
    test_accuracy, predictions, true_labels, report = evaluate_model_with_tta(model, test_loader, num_tta=5)
    
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
    
    print("增强训练完成！")

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
