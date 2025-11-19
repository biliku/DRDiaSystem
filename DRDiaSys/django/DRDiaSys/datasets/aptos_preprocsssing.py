import cv2
import numpy as np
import os
from tqdm import tqdm
import glob

# --- 图像处理参数 ---
TARGET_IMG_SIZE = 512

# --- 增强参数 (默认值, 会被 exp_ 系列参数覆盖) ---
GAUSSIAN_BLUR_SIGMA = 10
UNSHARP_WEIGHT = 1.5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8) # 明确写出CLAHE的网格大小

# --- 图像处理函数 ---
# (您的函数 apply_unsharp_mask_via_blur, apply_clahe, resize_with_padding 保持不变)
def apply_unsharp_mask_via_blur(image: np.ndarray, sigma: float, weight: float) -> np.ndarray:
    """
    通过高斯模糊实现非锐化掩蔽效果。
    sharpened = original + weight * (original - blurred)
    这个方法可以增强细节，并且如果原始背景是纯黑(0)，则处理后背景仍为纯黑(0)。
    """
    if image.ndim not in [2, 3]: # 支持灰度和彩色图
        # print(f"警告: 非锐化掩蔽不支持的图像形状: {image.shape}。返回原图。") # 避免过多打印
        return image.copy()

    img_float = image.astype(np.float32)

    if image.ndim == 3: # 彩色图像，在每个通道上操作
        channels = cv2.split(img_float)
        processed_channels = []
        for chan in channels:
            blurred_chan = cv2.GaussianBlur(chan, (0,0), sigmaX=sigma, sigmaY=sigma)
            sharpened_chan = chan + weight * (chan - blurred_chan)
            processed_channels.append(sharpened_chan)
        sharpened_img_float = cv2.merge(processed_channels)
    else: # 灰度图像
        blurred_img_float = cv2.GaussianBlur(img_float, (0,0), sigmaX=sigma, sigmaY=sigma)
        sharpened_img_float = img_float + weight * (img_float - blurred_img_float)

    # 将结果裁剪到 [0, 255] 范围并转换为 uint8
    sharpened_img_uint8 = np.clip(sharpened_img_float, 0, 255).astype(np.uint8)
    return sharpened_img_uint8

def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: tuple) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3: # RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)
        lab[:, :, 0] = cl
        processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1): # Grayscale
        l_channel = image.squeeze()
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        processed_image = clahe.apply(l_channel)
        if image.ndim == 3 and image.shape[2] == 1: #恢复通道维度
            processed_image = processed_image[:, :, np.newaxis]
    else:
        # print(f"警告: CLAHE不支持的图像形状: {image.shape}。返回原图。") # 避免过多打印
        return image.copy() # 返回副本以避免原地修改
    return processed_image

def resize_with_padding(image: np.ndarray, target_size: int, interpolation=cv2.INTER_AREA, pad_value=0) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        if image.ndim == 3: return np.full((target_size, target_size, image.shape[2]), pad_value, dtype=image.dtype)
        else: return np.full((target_size, target_size), pad_value, dtype=image.dtype)
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    if nh == 0: nh = 1 # 确保不为0
    if nw == 0: nw = 1 # 确保不为0
    resized = cv2.resize(image, (nw, nh), interpolation=interpolation) # 确保nw, nh > 0
    if image.ndim == 3: canvas = np.full((target_size, target_size, image.shape[2]), pad_value, dtype=image.dtype)
    else: canvas = np.full((target_size, target_size), pad_value, dtype=image.dtype)
    dh = (target_size - nh) // 2
    dw = (target_size - nw) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas

# --- 主处理逻辑 ---

def process_images_in_folder(
    source_folder: str,
    output_folder: str,
    target_img_size: int = TARGET_IMG_SIZE,
    blur_sigma_unsharp: float = GAUSSIAN_BLUR_SIGMA,
    unsharp_w: float = UNSHARP_WEIGHT,
    clahe_clip: float = CLAHE_CLIP_LIMIT,
    clahe_grid: tuple = CLAHE_TILE_GRID_SIZE, # 使用默认值
    supported_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
):
    if not os.path.isdir(source_folder):
        print(f"错误: 源文件夹 '{source_folder}' 不存在。")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"源文件夹: {os.path.abspath(source_folder)}")
    print(f"输出文件夹: {os.path.abspath(output_folder)}")
    print(f"参数: 目标尺寸={target_img_size}, 非锐化Sigma={blur_sigma_unsharp}, 非锐化权重={unsharp_w}, CLAHE限制={clahe_clip}, CLAHE网格={clahe_grid}")

    image_paths = []
    for ext in supported_extensions:
        # 收集所有可能的扩展名文件路径 (包括大写扩展名)
        image_paths.extend(glob.glob(os.path.join(source_folder, f'*{ext}')))
        image_paths.extend(glob.glob(os.path.join(source_folder, f'*{ext.upper()}')))

    # 使用集合去重，并排序以保证处理顺序一致性（如果需要）
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"在 '{source_folder}' 中未找到支持的图像扩展名: {supported_extensions}")
        return

    print(f"找到 {len(image_paths)} 张图像待处理。")

    for img_path in tqdm(image_paths, desc="图像预处理中"):
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"警告: 无法读取图像 {img_path}。跳过此图像。")
            continue

        # 1. BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. 应用非锐化掩蔽（通过高斯模糊）来增强细节
        img_sharpened = apply_unsharp_mask_via_blur(img_rgb, sigma=blur_sigma_unsharp, weight=unsharp_w)

        # 3. CLAHE 增强局部对比度
        img_clahe = apply_clahe(img_sharpened, clip_limit=clahe_clip, tile_grid_size=clahe_grid)

        # 4. Resize + padding: 将处理后的图像缩放并填充到目标尺寸的黑色画布上
        img_resized_padded = resize_with_padding(img_clahe, target_img_size, pad_value=0)

        # 5. RGB -> BGR for saving
        img_to_save_bgr = cv2.cvtColor(img_resized_padded, cv2.COLOR_RGB2BGR)

        save_file_path = os.path.join(output_folder, filename)
        try:
            cv2.imwrite(save_file_path, img_to_save_bgr)
        except Exception as e:
            print(f"保存图像 {save_file_path} 时出错: {e}")

    print(f"\n预处理完成。已处理（或尝试处理） {len(image_paths)} 张图像。")
    print(f"处理后的图像保存在: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: # 在交互式环境 (如Jupyter Notebook) 中运行时
        current_dir = os.getcwd()
        print(f"在交互模式下运行。使用当前工作目录: {current_dir}")

    # --- 用户配置区域 ---
    # !!! 请务必修改为您的实际路径 !!!
    SOURCE_IMAGES_FOLDER = os.path.join(current_dir, 'dataset', 'aptos2019', 'val_images') # 示例：原始图片文件夹
    PREPROCESSED_IMAGES_FOLDER = os.path.join(current_dir, 'dataset', 'aptos2019', 'val_images_processed') # 示例：处理后图片保存文件夹

    # --- 实验参数 ---
    # 修改这些值以调整预处理效果

    # 非锐化掩蔽的 Sigma (exp_blur_sigma_unsharp):
    #   - 5-10: 锐化更精细的细节，可能使图像看起来更“清晰”或“噪点更多”。
    #   - 10-20: 适用于一般细节增强的良好平衡点 (当前默认值为 10)。
    #   - 20-50: 增强更大尺度的细节，有助于提高相对于更平滑背景的整体对比度。
    exp_blur_sigma_unsharp = 20

    # 非锐化掩蔽的权重 (exp_unsharp_weight):
    #   - 0.8-1.2: 较温和的增强，颗粒感较少。
    #   - 1.3-1.8: 中等到较强的增强 (当前默认值为 1.5)。
    #   - 1.9-2.5 (或更高): 非常强的增强，更多颗粒感，可能出现光晕。
    exp_unsharp_weight = 1.2

    # CLAHE 的 Clip Limit (exp_clahe_clip):
    #   - 1.0-1.5: 较温和的局部对比度增强。
    #   - 2.0-3.0: 中等到较强的局部对比度增强 (当前默认值为 2.0)。
    #   - 3.0-5.0: 非常强，会显著改变外观并可能增加噪声。
    exp_clahe_clip = 2.0

    # CLAHE 的 Tile Grid Size (exp_clahe_grid_size):
    #   - (8,8) 是一个常见的默认值。
    #   - 较小的网格 (例如 (4,4)) 适应性更局部，可能更具侵略性。
    #   - 较大的网格 (例如 (16,16)) 局部性较差，行为更像全局直方图均衡化。
    exp_clahe_grid_size = (8, 8) # 之前是隐式(8,8)，现在明确指定

    if not os.path.isdir(SOURCE_IMAGES_FOLDER):
        print(f"错误: 源图像文件夹不存在: {os.path.abspath(SOURCE_IMAGES_FOLDER)}")
    else:
        print("--- 开始图像预处理 (可调非锐化掩蔽和CLAHE) ---")
        process_images_in_folder(
            source_folder=SOURCE_IMAGES_FOLDER,
            output_folder=PREPROCESSED_IMAGES_FOLDER,
            target_img_size=TARGET_IMG_SIZE,
            blur_sigma_unsharp=exp_blur_sigma_unsharp,
            unsharp_w=exp_unsharp_weight,
            clahe_clip=exp_clahe_clip,
            clahe_grid=exp_clahe_grid_size # 传递新的参数
        )
        print("--- 图像预处理脚本执行完毕 ---")
