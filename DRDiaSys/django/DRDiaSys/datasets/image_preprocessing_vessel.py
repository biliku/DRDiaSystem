import cv2
import numpy as np
import os
# from scipy import ndimage # 未使用，可删除
# from skimage import exposure # 未使用，可删除


class ImagePreprocessor:
    def __init__(self):
        # CLAHE参数：clipLimit可以根据需要调整。
        # 针对背景校正后的图像，可以尝试较低的clipLimit来更激进地增强血管。
        # 但如果背景校正已经很好了，clipLimit也可以稍高，以避免放大残余噪声。
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # 保持2.0，因为它在背景校正后效果可能更好

    def load_and_normalize(self, image):
        """
        数据加载与标准化
        """
        if image is None:
            raise ValueError("输入图像为空")
        # 确保图像是float32类型并归一化到0-1范围
        image = image.astype(np.float32) / 255.0
        return image
    
    def denoise(self, image):
        """
        图像去噪：采用轻度中值滤波，避免模糊细小血管
        """
        try:
            # 仅使用中值滤波，且核尺寸为3x3，这是非常轻度的去噪
            # 去除高斯滤波以减少模糊
            image_8bit = (image * 255).astype(np.uint8)
            denoised_8bit = cv2.medianBlur(image_8bit, 3) 
            return denoised_8bit.astype(np.float32) / 255.0
        except Exception as e:
            print(f"去噪过程出错: {str(e)}")
            return image
    
    def extract_green_channel(self, image):
        """
        提取绿色通道
        """
        try:
            if len(image.shape) == 3:
                # OpenCV 读取是 BGR 顺序，所以绿色通道是索引 1
                return image[:, :, 1]
            return image # 如果已经是灰度图，则直接返回
        except Exception as e:
            print(f"提取绿色通道出错: {str(e)}")
            return image
    
    def enhance_contrast(self, image):
        """
        对比度增强 (CLAHE)
        """
        try:
            # 将图像转换回0-255范围用于CLAHE
            image_8bit = (image * 255).astype(np.uint8)
            # 应用CLAHE
            enhanced = self.clahe.apply(image_8bit)
            # 转换回0-1范围
            enhanced = enhanced.astype(np.float32) / 255.0
            return enhanced
        except Exception as e:
            print(f"对比度增强出错: {str(e)}")
            return image

    def apply_mask(self, image, mask_path):
        """
        应用ROI Mask，mask为单通道0/1图像
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"未找到Mask文件: {mask_path}，跳过Mask处理")
            return image
        
        # 确保mask尺寸与图像匹配
        # 注意：这里假设输入图像和mask都是 HxW (灰度图)
        if mask.shape != image.shape:
            # print(f"警告: 图像和mask尺寸不匹配。图像尺寸: {image.shape}, Mask尺寸: {mask.shape}。进行resize。")
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 归一化mask到0-1 (0代表非ROI，255代表ROI)
        mask = mask.astype(np.float32) / 255.0
        
        # 应用mask，将区域外像素置为0 (黑色)
        masked_image = image * mask
        return masked_image

    def process_image(self, image, mask_path=None):
        """
        完整的图像预处理流程 (针对血管分割优化，适用于暗血管、暗病灶图像)
        """
        try:
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    raise ValueError(f"无法读取图像文件: {image}")

            # 1. 数据加载与标准化 (原始图像归一化)
            normalized_image = self.load_and_normalize(image)

            # 2. 提取绿色通道 / 确定单通道图像
            # 如果输入已经是灰度图（如您提供的“原始图像”），则直接使用它
            if len(normalized_image.shape) == 3:
                current_image = self.extract_green_channel(normalized_image)
            else:
                current_image = normalized_image # 已经是灰度图

            # 3. 轻度去噪 (使用3x3中值滤波，去除高斯滤波以减少模糊)
            current_image = self.denoise(current_image)

            # 4. 背景校正 (Background Subtraction / Illumination Normalization)
            # 目的：估计图像的低频背景（包括光照不均和大面积弥散病灶），然后减去它，
            # 以突出高频细节（如血管）并压制大面积病灶。
            # kernel_size: 应远大于血管宽度，但可能小于最大的弥散病灶，以有效去除它们。
            # 常见值：51, 101, 151等。这里选择 101。
            background_kernel_size = 101 
            background = cv2.GaussianBlur(current_image, (background_kernel_size, background_kernel_size), 0)
            
            # 从原图（或去噪图）中减去背景。
            # 这样，比局部背景暗的区域（如血管）会变得更暗，而大面积的暗病灶则会被“抬升”。
            # 由于图像是0-1范围，直接相减可能产生负值，需要后续归一化。
            corrected_image = current_image - background
            
            # 归一化到0-1范围，将图像拉伸到完整动态范围。
            # 这会将血管（负值）拉伸到0附近，背景（0或正值）拉伸到1附近，从而增强血管对比度。
            corrected_image = cv2.normalize(corrected_image, None, 0, 1, cv2.NORM_MINMAX)

            # 5. 对比度增强 (CLAHE)
            # 在背景校正后，CLAHE可以进一步增强血管的局部对比度。
            final_processed_image = self.enhance_contrast(corrected_image)
            
            # 6. 可选：应用ROI Mask (确保眼底外部区域为黑色)
            if mask_path:
                final_processed_image = self.apply_mask(final_processed_image, mask_path)

            return final_processed_image
        except Exception as e:
            print(f"图像处理过程中出现错误: {str(e)}")
            return None

    def save_processed_image(self, image, output_path):
        """
        保存处理后的图像
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 将图像转换回0-255范围
            image_8bit = (image * 255).astype(np.uint8)
            # 保存图像
            cv2.imwrite(output_path, image_8bit)
            print(f"图像已保存到: {output_path}")
        except Exception as e:
            print(f"保存图像时出错: {str(e)}")

    def display_images(self, input_path, mask_path=None):
        """
        显示原始图像和处理后的图像
        """
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"错误：无法读取图像文件 {input_path}")
                return
            
            processed_image = self.process_image(image, mask_path)
            
            if processed_image is not None:
                cv2.imshow("Original", image)
                cv2.imshow("Processed", (processed_image * 255).astype(np.uint8))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("图像处理失败")
        except Exception as e:
            print(f"显示图像时出错: {str(e)}")

def process_batch(images):
    """
    批量处理图像 (此函数通常用于内存中图像，如果是文件建议使用 process_directory)
    """
    preprocessor = ImagePreprocessor()
    processed_images = []
    
    for i, image in enumerate(images):
        try:
            processed = preprocessor.process_image(image) # 这里没有mask_path，如果需要，需修改传入方式
            if processed is not None:
                processed_images.append(processed)
            else:
                print(f"处理第 {i+1} 张图像失败")
        except Exception as e:
            print(f"处理第 {i+1} 张图像时出现错误: {str(e)}")
    
    return np.array(processed_images)

def process_directory(input_dir, output_dir, mask_dir=None):
    """
    批量处理目录下所有图片，并保存到输出目录
    mask_dir: mask文件夹路径（可选），如有则自动匹配mask
    """
    processor = ImagePreprocessor()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if not os.path.isfile(input_path): # 确保是文件而不是目录
                continue
            
            image = cv2.imread(input_path)
            if image is None:
                print(f"无法读取图像: {input_path}")
                continue
            
            mask_path = None
            if mask_dir:
                name, ext = os.path.splitext(filename)
                # 尝试常见的mask命名规则 (根据你的数据集调整)
                mask_candidates = [
                    os.path.join(mask_dir, f"{name}_mask.png"),
                    os.path.join(mask_dir, f"{name}.png"),
                    os.path.join(mask_dir, f"{name}_mask.gif"),
                    os.path.join(mask_dir, f"{name}.gif")
                ]
                for candidate in mask_candidates:
                    if os.path.exists(candidate):
                        mask_path = candidate
                        break
                if mask_path is None:
                    print(f"未找到对应mask: {name}，跳过mask处理")
            
            processed = processor.process_image(image, mask_path)
            if processed is not None:
                processor.save_processed_image(processed, output_path)
            else:
                print(f"处理失败: {input_path}")


if __name__ == "__main__":
    # 请根据你的实际路径修改以下变量
    input_images_dir = "F:\DRDiaSys\django\DRDiaSys\datasets\dataset\DRIVE\\test\images"
    output_processed_dir = "F:\\DRDiaSys\\django\\DRDiaSys\\datasets\\dataset\\DRIVE\\test\\processed_images" # 新的输出目录
    mask_files_dir = "F:\\DRDiaSys\\django\\DRDiaSys\\datasets\\dataset\\DRIVE\\test\\mask"

    print(f"开始处理图像从: {input_images_dir}")
    print(f"处理后的图像将保存到: {output_processed_dir}")
    print(f"Mask文件目录: {mask_files_dir}")

    process_directory(
        input_images_dir,
        output_processed_dir,
        mask_dir=mask_files_dir
    )
    print("所有图像处理完成。")

    # 可以单独测试一张图像，并显示效果
    # 假设你的DRIVE数据集图片是 .tif 格式，mask是 .gif 格式
    # 如果你的"问题图像"是DRIVE数据集中的，比如 05_training.tif (因为它有弥散病灶)
    test_problem_image_path = os.path.join(input_images_dir, "05_training.tif")
    test_problem_mask_path = os.path.join(mask_files_dir, "05_training_mask.gif") 
    
    if os.path.exists(test_problem_image_path) and os.path.exists(test_problem_mask_path):
        print(f"\n正在测试病灶图像: {test_problem_image_path}")
        preprocessor = ImagePreprocessor()
        preprocessor.display_images(test_problem_image_path, test_problem_mask_path)
    else:
        print(f"\n测试图像或Mask文件不存在。请检查路径: {test_problem_image_path} 和 {test_problem_mask_path}")

