import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

# --- 全局配置 ---
TARGET_IMG_SIZE = 512

# --- 核心预处理函数 ---

def circle_crop_and_clean(img_bgr: np.ndarray, sigma_x: float = 10) -> np.ndarray:
    """
    对图像进行高质量的圆形裁剪，去除黑边和伪影。
    这是预处理中最关键的一步。
    """
    # 1. 颜色转换至RGB，这是更标准的处理空间
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 初步裁剪，移除最外层的纯黑边
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray_img > 7 # 容忍度，去除接近纯黑的像素
    if not np.any(mask): # 如果图像是全黑的
        return np.zeros((TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3), dtype=np.uint8)
        
    img_rgb_cropped = img_rgb[np.ix_(mask.any(1), mask.any(0))]

    # 3. 在裁剪后的图像上进行高斯模糊，以平滑图像，便于找到圆形轮廓
    # 使用原始图像的副本进行模糊，避免在原始清晰图像上操作
    img_blurred = cv2.GaussianBlur(img_rgb_cropped, (0, 0), sigma_x)

    # 4. 创建圆形掩码并应用
    height, width, _ = img_rgb_cropped.shape
    center_x, center_y = width // 2, height // 2
    radius = min(center_x, center_y)

    mask_circle = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_circle, (center_x, center_y), radius, 1, thickness=-1)

    # 将掩码应用到清晰的（未模糊的）裁剪图像上
    img_masked = cv2.bitwise_and(img_rgb_cropped, img_rgb_cropped, mask=mask_circle)
    
    # 5. 最后一次裁剪，移除圆形裁剪后新产生的黑角
    gray_masked = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY)
    final_mask = gray_masked > 7
    if not np.any(final_mask):
         return np.zeros((TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3), dtype=np.uint8)
    
    final_image = img_masked[np.ix_(final_mask.any(1), final_mask.any(0))]
    
    return final_image

def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    将干净的、裁剪后的图像缩放到目标尺寸。
    """
    # 使用 INTER_AREA 插值，它在缩小图像时效果最好，能最大限度保留信息
    return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)


def preprocess_image_pipeline(source_path: str, target_size: int):
    """
    完整的加载、处理、保存流程。
    """
    img_bgr = cv2.imread(source_path)
    if img_bgr is None:
        print(f"警告: 无法读取图像 {source_path}。")
        return None

    # 步骤 1: 圆形裁剪和清洁 (核心)
    cleaned_image_rgb = circle_crop_and_clean(img_bgr)

    # 步骤 2: 调整尺寸
    resized_image_rgb = resize_image(cleaned_image_rgb, target_size)

    # 步骤 3: 转回 BGR 以便保存
    final_image_bgr = cv2.cvtColor(resized_image_rgb, cv2.COLOR_RGB2BGR)
    
    return final_image_bgr


# --- 主执行逻辑 ---
def run_preprocessing(source_folder: str, output_folder: str, target_img_size: int):
    os.makedirs(output_folder, exist_ok=True)
    print(f"源文件夹: {os.path.abspath(source_folder)}")
    print(f"输出文件夹: {os.path.abspath(output_folder)}")

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(os.path.join(source_folder, f'**/*{ext}'), recursive=True))

    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"在 '{source_folder}' 中未找到支持的图像。")
        return

    print(f"找到 {len(image_paths)} 张图像待处理。")

    for img_path in tqdm(image_paths, desc="图像预处理中"):
        filename = os.path.basename(img_path)
        
        processed_img = preprocess_image_pipeline(img_path, target_img_size)
        
        if processed_img is not None:
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, processed_img)
            
    print(f"\n预处理完成！处理后的图像保存在: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    # --- ★★★ 用户配置区 ★★★ ---
    # !!! 请务必修改为您的实际路径 !!!
    # 使用一个全新的输出文件夹，以避免与旧的预处理结果混淆
    SOURCE_IMAGES_FOLDER = 'F:\\DRDiaSys\\django\\DRDiaSys\\datasets\\dataset\\aptos2019\\val_images' 
    PREPROCESSED_IMAGES_FOLDER = 'F:\\DRDiaSys\\django\\DRDiaSys\\datasets\\dataset\\aptos2019\\val_images_processed_v2' 
    # --- ★★★ 配置结束 ★★★ ---

    if not os.path.isdir(SOURCE_IMAGES_FOLDER):
        print(f"错误: 源图像文件夹不存在: {os.path.abspath(SOURCE_IMAGES_FOLDER)}")
    else:
        run_preprocessing(
            source_folder=SOURCE_IMAGES_FOLDER,
            output_folder=PREPROCESSED_IMAGES_FOLDER,
            target_img_size=TARGET_IMG_SIZE
        )

