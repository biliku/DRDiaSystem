import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2 # 用于后处理

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF # 用于同步变换

# ----------- U-Net结构 -----------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # bias=False since using BN
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.middle = DoubleConv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512) # 512 (from up4) + 512 (from d4)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256) # 256 (from up3) + 256 (from d3)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128) # 128 (from up2) + 128 (from d2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)   # 64 (from up1) + 64 (from d1)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        # Bottleneck
        m = self.middle(p4)

        # Decoder
        u4 = self.up4(m)
        # Pad u4 if its H, W are not equal to d4's H, W (due to non-even input size before pooling)
        # This can happen if input_size is not perfectly divisible by 2^4
        if u4.shape[2:] != d4.shape[2:]:
            u4 = TF.resize(u4, size=d4.shape[2:])
        c4 = torch.cat([u4, d4], dim=1)
        c4 = self.conv4(c4)

        u3 = self.up3(c4)
        if u3.shape[2:] != d3.shape[2:]:
            u3 = TF.resize(u3, size=d3.shape[2:])
        c3 = torch.cat([u3, d3], dim=1)
        c3 = self.conv3(c3)

        u2 = self.up2(c3)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = TF.resize(u2, size=d2.shape[2:])
        c2 = torch.cat([u2, d2], dim=1)
        c2 = self.conv2(c2)

        u1 = self.up1(c2)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = TF.resize(u1, size=d1.shape[2:])
        c1 = torch.cat([u1, d1], dim=1)
        c1 = self.conv1(c1)
        
        out = self.out_conv(c1)
        return out

# ----------- 数据集 -----------
class VesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_pairs, image_size=(512, 512), is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.file_pairs = file_pairs # List of (img_name, mask_name) tuples
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.file_pairs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # 图像大小调整
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST) # Mask用最近邻插值

        # 数据增强 (仅在训练时)
        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 随机垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            # 随机旋转
            angle = random.choice([0, 90, 180, 270]) # 或者 random.uniform(-30, 30)
            if angle != 0:
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            # 可以添加更多增强:
            # 如 ColorJitter (先转RGB再转回L，或直接对L操作，需小心)
            # image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)

        # 转换为Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # 确保掩码是二值的 (0 或 1)
        mask = (mask > 0.5).float() # PIL读取的L模式是0-255，ToTensor后是0-1
        
        return image, mask

# ----------- 损失函数 -----------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs) # 如果模型输出logits
        
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / \
                     (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1 - dice_coeff

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, bce_pos_weight=None): # alpha是BCE的权重
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        loss_bce = self.bce(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        return self.alpha * loss_bce + (1 - self.alpha) * loss_dice

# ----------- 训练主流程 -----------
def train():
    # --- 参数配置 ---
    IMG_DIR = 'F:\DRDiaSys\django\DRDiaSys\datasets\dataset\DRIVE\\training\processed_images_v4' # 使用你最好的预处理结果
    MASK_DIR = 'F:\DRDiaSys\django\DRDiaSys\datasets\dataset\DRIVE\\training\\1st_manual'
    MODEL_SAVE_PATH = 'unet_vessel_segmentation.pth'
    
    IMAGE_SIZE = (512, 512) # U-Net通常要求输入尺寸是2的N次方，以避免维度不匹配
    EPOCHS = 100  # 增加epoch数量，配合早停
    BATCH_SIZE = 4 # 根据显存调整，如果用512x512，batch可能要小
    LR = 1e-4      # 初始学习率，可以尝试1e-3, 3e-4, 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VAL_SPLIT = 0.2 # 20%的数据作为验证集
    EARLY_STOPPING_PATIENCE = 10 # 如果验证损失连续10个epoch没有改善，则停止
    # --- 结束参数配置 ---

    print(f"使用设备: {DEVICE}")

    # 准备文件对 (图像名, 掩码名)
    all_img_names = sorted(os.listdir(IMG_DIR))
    all_mask_names = sorted(os.listdir(MASK_DIR))
    
    mask_map = {m[:2]: m for m in all_mask_names} # 假设掩码名前两个字符与图像对应
    
    paired_files = []
    for img_name in all_img_names:
        key = img_name[:2]
        if key in mask_map:
            paired_files.append((img_name, mask_map[key]))
        else:
            print(f"警告: 图像 {img_name} 没有找到对应的掩码，将被跳过。")

    if not paired_files:
        print("错误: 没有找到任何图像-掩码对。请检查路径和文件名格式。")
        return

    # 划分训练集和验证集
    random.shuffle(paired_files) # 打乱顺序
    split_idx = int(len(paired_files) * (1 - VAL_SPLIT))
    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    print(f"总样本数: {len(paired_files)}, 训练样本数: {len(train_files)}, 验证样本数: {len(val_files)}")

    train_dataset = VesselDataset(IMG_DIR, MASK_DIR, train_files, image_size=IMAGE_SIZE, is_train=True)
    val_dataset = VesselDataset(IMG_DIR, MASK_DIR, val_files, image_size=IMAGE_SIZE, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 模型、损失、优化器
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    
    # 可选: 计算BCE的pos_weight (如果需要)
    # pos_weight_tensor = calculate_bce_pos_weight(train_dataset, DEVICE)
    # criterion = ComboLoss(alpha=0.5, bce_pos_weight=pos_weight_tensor).to(DEVICE)
    criterion = ComboLoss(alpha=0.5).to(DEVICE) # alpha=0.5表示BCE和Dice权重相同

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for images, masks in train_progress_bar:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_train_loss / len(train_loader)

        # 验证
        model.eval()
        epoch_val_loss = 0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for images, masks in val_progress_bar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                val_progress_bar.set_postfix(loss=loss.item())
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss) # 基于验证损失调整学习率

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型已保存到: {MODEL_SAVE_PATH}")


# ----------- 推理与可视化 -----------
def predict(model_path, img_path, save_path=None, image_size=(512, 512), use_postprocessing=True):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"错误: 模型文件未找到 {model_path}")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
        
    model.eval()
    
    try:
        image_pil = Image.open(img_path).convert('L')
    except FileNotFoundError:
        print(f"错误: 输入图像文件未找到 {img_path}")
        return
    except Exception as e:
        print(f"打开图像时出错: {e}")
        return

    # 预处理: 调整大小 + ToTensor (不进行训练时的随机增强)
    image_resized = TF.resize(image_pil, image_size)
    input_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(DEVICE) # (1, 1, H, W)
    
    with torch.no_grad():
        output = model(input_tensor) # (1, 1, H, W)
        pred_probs = torch.sigmoid(output).squeeze().cpu().numpy() # (H, W), 0-1范围
        
    # 阈值化得到二值掩码
    # pred_mask_binary = (pred_probs > 0.5).astype(np.uint8) 
        pred_mask_binary = (pred_probs > 0.22).astype(np.uint8) # 尝试不同的值，如0.4, 0.35, 0.3

    
    # 后处理 (可选)
    if use_postprocessing:
        print("应用温和的后处理...")
        # 主要使用闭运算来连接断点
        # 可以尝试不同的核大小和形状
        kernel_close_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) # 尝试(3,3) 或 (5,5)
        # kernel_close_small = np.ones((3,3), np.uint8) # 或者方形核
        
        pred_mask_binary = cv2.morphologyEx(pred_mask_binary, cv2.MORPH_CLOSE, kernel_close_small, iterations=1) # iterations也可以调整

        # （可选）非常非常谨慎地使用小核的开运算去除极端噪点，但要小心别又把细血管去掉了
        # kernel_open_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)) # 极小的核
        # if kernel_open_tiny.sum() > 0: # 确保核不为空
        #    pred_mask_binary = cv2.morphologyEx(pred_mask_binary, cv2.MORPH_OPEN, kernel_open_tiny, iterations=1)


    # 将二值掩码转换为0-255范围的图像
    pred_mask_display = pred_mask_binary * 255 
    
    # 缩放回原始图像大小 (如果需要)
    # original_size = image_pil.size # (width, height)
    # pred_mask_display = cv2.resize(pred_mask_display, original_size, interpolation=cv2.INTER_NEAREST)

    # 保存或显示
    if save_path:
        try:
            Image.fromarray(pred_mask_display).save(save_path)
            print(f"预测结果已保存到: {save_path}")
        except Exception as e:
            print(f"保存预测结果时出错: {e}")
    else:
        Image.fromarray(pred_mask_display).show()


# (可选) 计算BCE损失的pos_weight的辅助函数
# def calculate_bce_pos_weight(dataset, device):
#     num_pos = 0
#     num_total = 0
#     print("正在计算BCE pos_weight...")
#     for _, mask in tqdm(dataset): # 遍历一次数据集来统计
#         num_pos += mask.sum()
#         num_total += mask.numel()
#     if num_pos == 0: return torch.tensor([1.0]).to(device) # 避免除以零
#     num_neg = num_total - num_pos
#     pos_weight = num_neg / num_pos
#     print(f"计算得到 pos_weight: {pos_weight:.4f}")
#     return torch.tensor([pos_weight]).to(device)


if __name__ == '__main__':
    # --- 训练 ---
    # train()
    
    # --- 推理示例 ---
    # (确保训练完成后模型文件 'unet_vessel_segmentation.pth' 存在)
    
    # 测试图像路径 (请替换为你的测试图像路径)
    test_image_preprocessed_example = r'F:\DRDiaSys\django\DRDiaSys\datasets\dataset\IDRiD\1. Original Images\a. Training Set\IDRiD_02.jpg' # 假设你对测试图也做了预处理
    result_save_path = r'F:\DRDiaSys\vessel_segmentation_result\\result_newmodel6.png'

    print(f"\n开始对图像 {test_image_preprocessed_example} 进行推理...")
    predict(
        model_path='unet_vessel_segmentation.pth', # 训练好的模型
        img_path=test_image_preprocessed_example,   # 输入处理后的图像
        save_path=result_save_path,
        image_size=(512, 512), # 与训练时一致
        use_postprocessing=False
    )
