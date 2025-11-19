import cv2
import numpy as np
import os

class ImagePreprocessor:
    def __init__(self,
                 clahe_clip_limit=2.0,
                 clahe_tile_grid_size=(8, 8),
                 background_kernel_size=101, # 用于背景估计的高斯模糊核
                 denoise_median_kernel_size=3,
                 final_gaussian_kernel_size=0,
                 gamma=1.0,  # 新增：伽马校正值，1.0表示不校正
                 use_morphological_enhancement=True, # 新增：是否使用形态学增强
                 morph_kernel_size=21, # 新增：形态学操作（如底帽）的核大小，应略大于最粗血管直径
                 morph_op_iterations=1): # 新增：形态学操作迭代次数
        
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        self.background_kernel_size = background_kernel_size
        self.denoise_median_kernel_size = denoise_median_kernel_size
        self.final_gaussian_kernel_size = final_gaussian_kernel_size
        self.gamma = gamma
        self.use_morphological_enhancement = use_morphological_enhancement
        self.morph_kernel_size = morph_kernel_size # 确保是奇数
        if self.morph_kernel_size % 2 == 0:
            self.morph_kernel_size +=1
        self.morph_op_iterations = morph_op_iterations


    def load_and_normalize(self, image):
        if image is None:
            raise ValueError("输入图像为空")
        image = image.astype(np.float32) / 255.0
        return image

    def gamma_correction(self, image):
        """
        伽马校正。gamma < 1.0 会使图像变亮，gamma > 1.0 会使图像变暗。
        """
        if self.gamma == 1.0: # 1.0 表示不进行校正
            return image
        # 构建lookup table
        inv_gamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        
        # 应用lookup table
        try:
            if np.issubdtype(image.dtype, np.floating): # 如果是0-1浮点
                image_for_gamma = (image * 255).astype(np.uint8)
                gamma_corrected_8bit = cv2.LUT(image_for_gamma, table)
                return gamma_corrected_8bit.astype(np.float32) / 255.0
            elif image.dtype == np.uint8: # 如果已经是8位
                return cv2.LUT(image, table)
            else:
                print(f"伽马校正暂不支持的图像类型: {image.dtype}")
                return image
        except Exception as e:
            print(f"伽马校正过程出错: {str(e)}")
            return image

    def denoise(self, image):
        if self.denoise_median_kernel_size <= 1 or self.denoise_median_kernel_size % 2 == 0:
            return image
        try:
            # 确保输入是8位图进行中值滤波
            if np.issubdtype(image.dtype, np.floating):
                image_8bit = (image * 255).astype(np.uint8)
                denoised_8bit = cv2.medianBlur(image_8bit, self.denoise_median_kernel_size)
                return denoised_8bit.astype(np.float32) / 255.0
            elif image.dtype == np.uint8:
                return cv2.medianBlur(image, self.denoise_median_kernel_size)
            else:
                print(f"去噪暂不支持的图像类型: {image.dtype}")
                return image
        except Exception as e:
            print(f"去噪过程出错: {str(e)}")
            return image

    def extract_green_channel(self, image):
        try:
            if len(image.shape) == 3 and image.shape[2] == 3: # 确保是彩色图
                return image[:, :, 1]
            return image
        except Exception as e:
            print(f"提取绿色通道出错: {str(e)}")
            return image

    def enhance_contrast_clahe(self, image): # 重命名以区分
        try:
            # CLAHE 需要8位图输入
            if np.issubdtype(image.dtype, np.floating):
                image_8bit = (image * 255).astype(np.uint8)
                enhanced_8bit = self.clahe.apply(image_8bit)
                return enhanced_8bit.astype(np.float32) / 255.0
            elif image.dtype == np.uint8:
                return self.clahe.apply(image)
            else:
                print(f"CLAHE暂不支持的图像类型: {image.dtype}")
                return image
        except Exception as e:
            print(f"对比度增强出错: {str(e)}")
            return image

    def enhance_vessels_morphologically(self, image):
        """
        使用形态学底帽变换增强血管等暗色细节。
        核大小应略大于目标血管的宽度。
        """
        if not self.use_morphological_enhancement or self.morph_kernel_size <=1:
            return image
        try:
            # 形态学操作通常在8位图上进行
            if np.issubdtype(image.dtype, np.floating):
                image_8bit = (image * 255).astype(np.uint8)
            elif image.dtype == np.uint8:
                image_8bit = image
            else:
                print(f"形态学增强暂不支持的图像类型: {image.dtype}")
                return image

            # 使用近似圆形的结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morph_kernel_size, self.morph_kernel_size))
            # 底帽变换 (Closing - Image) 提取暗色结构
            # closing = cv2.morphologyEx(image_8bit, cv2.MORPH_CLOSE, kernel, iterations=self.morph_op_iterations)
            # bottom_hat = cv2.subtract(closing, image_8bit) # OpenCV的subtract会自动处理溢出到0

            # 或者直接使用 cv2.MORPH_BLACKHAT (Image - Opening)
            # 对于灰度图，黑帽(image - opening(image)) 和 底帽(closing(image) - image) 效果相似，
            # 但黑帽更直接提取暗结构，底帽间接通过填充亮结构再相减。
            # 通常黑帽 (cv2.MORPH_BLACKHAT) 更适合直接提取血管这类暗结构。
            # 我们这里使用 黑帽 (Black-Hat)
            black_hat = cv2.morphologyEx(image_8bit, cv2.MORPH_BLACKHAT, kernel, iterations=self.morph_op_iterations)
            
            # 形态学变换后的结果也需要归一化，因为它本身就是一种对比度拉伸
            enhanced_morph = cv2.normalize(black_hat, None, 0, 255, cv2.NORM_MINMAX)

            if np.issubdtype(image.dtype, np.floating):
                return enhanced_morph.astype(np.float32) / 255.0
            else:
                return enhanced_morph
        except Exception as e:
            print(f"形态学增强出错: {str(e)}")
            return image

    def apply_mask(self, image, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"未找到Mask文件: {mask_path}，跳过Mask处理")
            return image
        
        if mask.shape[0:2] != image.shape[0:2]: # 比较高和宽
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 确保mask是二值的 (0或1 for float, 0 or 255 for uint8)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        if np.issubdtype(image.dtype, np.floating):
            mask_norm = mask_binary.astype(np.float32) / 255.0
            masked_image = image * mask_norm
        elif image.dtype == np.uint8:
            masked_image = cv2.bitwise_and(image, image, mask=mask_binary)
        else:
            print(f"Mask应用暂不支持的图像类型: {image.dtype}")
            return image
            
        return masked_image

    def process_image(self, image_input, mask_path=None):
        try:
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"无法读取图像文件: {image_input}")
            else: # 假设输入是 numpy array
                image = image_input.copy()


            # 1. 数据加载与标准化 (原始图像归一化到0-1 float)
            # 注意：后续很多opencv函数更喜欢uint8, 所以我们会在需要时转换
            # 为了流程统一，我们先转为0-1 float，然后在各步骤内部按需转uint8再转回float
            current_image_float = self.load_and_normalize(image)

            # 2. 提取绿色通道 (如果原图是彩色)
            current_image_float = self.extract_green_channel(current_image_float)
            
            # 3. (可选) 伽马校正 (在0-1 float上操作，内部会转uint8再转回)
            current_image_float = self.gamma_correction(current_image_float)

            # 4. 轻度去噪 (中值滤波，同样在0-1 float上操作)
            current_image_float = self.denoise(current_image_float)

            # --- 背景校正 和 形态学增强 的组合策略 ---
            # 策略1: 先进行背景扣除，然后在结果上进行形态学增强 (推荐)
            # 策略2: 直接对去噪图进行形态学增强 (如果背景本身较均匀)

            # 5. 背景校正 (Illumination Normalization)
            # 目的：估计并减去低频背景，突出血管等高频结构。
            # 确保输入是 0-1 float
            background = cv2.GaussianBlur(current_image_float,
                                        (self.background_kernel_size, self.background_kernel_size),
                                        0)
            corrected_image_float = current_image_float - background
            # 将减法结果重新归一化到0-1，这是关键一步，能拉伸对比度
            corrected_image_float = cv2.normalize(corrected_image_float, None, 0, 1, cv2.NORM_MINMAX)

            # 6. (可选) 形态学增强 (如黑帽/底帽变换)
            # 对背景校正后的图像进行操作，进一步锐化和提取血管
            if self.use_morphological_enhancement:
                # 形态学操作最好在对比度较好的图像上进行
                # 输入 corrected_image_float (0-1范围)
                enhanced_morph_float = self.enhance_vessels_morphologically(corrected_image_float)
                # 形态学变换（如黑帽）的结果本身就是增强后的血管，其灰度值分布可能已经适合CLAHE
                # 通常黑帽结果是血管亮，背景暗。但我们这里目标是血管暗，背景亮。
                # 如果 enhance_vessels_morphologically 返回的是血管亮，背景暗，需要反转
                # 但我们的 black_hat 提取暗结构，然后normalize了，所以应该是血管为高值（亮）
                # 为了与后续CLAHE以及期望的输出（血管暗）一致，这里可以考虑是否需要反转。
                # 在此实现中，enhance_vessels_morphologically已输出血管较亮、背景暗且归一化的图像。
                # 如果血管分割模型期望血管暗，背景亮，那么在形态学增强后可能需要反转
                # 或者调整形态学操作，例如：image - black_hat(image)
                # 鉴于原始流程的目标是血管更暗，这里我们假设形态学增强后，血管区域被提取为“前景”
                # 如果你的`enhance_vessels_morphologically`后血管是亮的，可以：
                # current_image_for_clahe = 1.0 - enhanced_morph_float # 反转，使血管变暗
                # 或者，如果你的形态学输出就是血管暗：
                current_image_for_clahe = enhanced_morph_float
            else:
                current_image_for_clahe = corrected_image_float


            # 7. 对比度增强 (CLAHE)
            # 应用于背景校正和（可选的）形态学增强之后的结果
            final_processed_image_float = self.enhance_contrast_clahe(current_image_for_clahe)

            # 8. 可选：最终的轻度高斯平滑
            if self.final_gaussian_kernel_size > 1 and self.final_gaussian_kernel_size % 2 == 1:
                # 输入 float, 输出 float
                image_8bit = (final_processed_image_float * 255).astype(np.uint8)
                smoothed_8bit = cv2.GaussianBlur(image_8bit,
                                                (self.final_gaussian_kernel_size, self.final_gaussian_kernel_size),
                                                0)
                final_processed_image_float = smoothed_8bit.astype(np.float32) / 255.0

            # 9. 应用ROI Mask
            if mask_path:
                # 输入 float, 输出 float
                final_processed_image_float = self.apply_mask(final_processed_image_float, mask_path)

            return final_processed_image_float
        except Exception as e:
            print(f"图像处理过程中出现错误 ({image_input if isinstance(image_input, str) else 'Numpy_array'}): {str(e)}")
            # 可以考虑在这里打印更详细的traceback
            import traceback
            traceback.print_exc()
            return None

    def save_processed_image(self, image, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 确保图像在0-1范围，然后转为0-255 uint8
            if np.issubdtype(image.dtype, np.floating):
                # 裁剪以防万一有值超出0-1范围
                image = np.clip(image, 0.0, 1.0)
                image_8bit = (image * 255).astype(np.uint8)
            elif image.dtype == np.uint8:
                image_8bit = image
            else:
                print(f"保存图像时不支持的类型 {image.dtype}, 尝试转换为float后处理")
                image_8bit = (image.astype(np.float32) * 255).astype(np.uint8)

            cv2.imwrite(output_path, image_8bit)
            print(f"图像已保存到: {output_path}")
        except Exception as e:
            print(f"保存图像时出错: {str(e)}")

    def display_images(self, input_path, mask_path=None):
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"错误：无法读取图像文件 {input_path}")
                return
            
            processed_image_float = self.process_image(image, mask_path)
            
            if processed_image_float is not None:
                # 确保显示时是uint8
                processed_image_uint8 = (np.clip(processed_image_float, 0.0, 1.0) * 255).astype(np.uint8)
                
                # 为了对比，显示原始的绿色通道（如果适用）
                original_green_float = self.extract_green_channel(self.load_and_normalize(image))
                original_green_uint8 = (np.clip(original_green_float, 0.0, 1.0) * 255).astype(np.uint8)

                cv2.imshow("Original (Green Channel or Grayscale)", original_green_uint8)
                cv2.imshow("Processed", processed_image_uint8)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("图像处理失败，无法显示")
        except Exception as e:
            print(f"显示图像时出错: {str(e)}")


def process_batch(images_list, mask_paths_list=None, **preprocessor_params): # 允许传入mask列表
    preprocessor = ImagePreprocessor(**preprocessor_params)
    processed_images_list = []
    
    for i, image_data in enumerate(images_list):
        mask_path = mask_paths_list[i] if mask_paths_list and i < len(mask_paths_list) else None
        try:
            processed = preprocessor.process_image(image_data, mask_path=mask_path) # 传入mask_path
            if processed is not None:
                processed_images_list.append(processed)
            else:
                print(f"处理第 {i+1} 张图像失败 (来自列表)")
        except Exception as e:
            print(f"处理第 {i+1} 张图像 (来自列表) 时出现错误: {str(e)}")
    
    return np.array(processed_images_list)


def process_directory(input_dir, output_dir, mask_dir=None, **preprocessor_params):
    processor = ImagePreprocessor(**preprocessor_params)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename) # 保留原文件名和扩展名
            
            if not os.path.isfile(input_path):
                continue
            
            # image = cv2.imread(input_path) # 在 process_image 内部读取
            # if image is None:
            #     print(f"无法读取图像: {input_path}")
            #     continue
            
            mask_path = None
            if mask_dir:
                name, ext = os.path.splitext(filename)
                # 更灵活的mask查找，可以添加更多扩展名
                mask_name_variations = [f"{name}_mask", name, f"{name.split('_')[0]}_mask"] # 比如 xx_training.tif -> xx_mask.gif
                mask_extensions = ['.png', '.gif', '.tif', '.jpg', '.bmp']
                
                found_mask = False
                for mn_var in mask_name_variations:
                    for m_ext in mask_extensions:
                        candidate = os.path.join(mask_dir, f"{mn_var}{m_ext}")
                        if os.path.exists(candidate):
                            mask_path = candidate
                            found_mask = True
                            break
                    if found_mask:
                        break
            
            # process_image 现在自己处理路径或numpy数组
            processed = processor.process_image(input_path, mask_path) 
            if processed is not None:
                processor.save_processed_image(processed, output_path)
            else:
                print(f"处理失败: {input_path}")


if __name__ == "__main__":
    input_images_dir = "F:\DRDiaSys\django\DRDiaSys\datasets\dataset\DRIVE\\test\images"
    output_processed_dir = "F:\DRDiaSys\django\DRDiaSys\datasets\dataset\DRIVE\\test\processed_images_v2" 
    mask_files_dir = "F:\DRDiaSys\django\DRDiaSys\datasets\dataset\DRIVE\\test\mask"

    print(f"开始处理图像从: {input_images_dir}")
    print(f"处理后的图像将保存到: {output_processed_dir}")
    if mask_files_dir:
        print(f"Mask文件目录: {mask_files_dir}")
    else:
        print("不使用Mask文件。")


    # --- 核心微调参数配置 ---
    # 针对 DRIVE 数据集，血管相对清晰，背景病灶不严重时
    # background_kernel_size 不需要特别大
    # morph_kernel_size 要根据血管粗细调整
    drive_preprocessor_config = {
        'clahe_clip_limit': 2.0,         
        'clahe_tile_grid_size': (8, 8),    
        'background_kernel_size': 151,   # 101-201 适合DRIVE，用于平滑背景光照
        'denoise_median_kernel_size': 3,   
        'final_gaussian_kernel_size': 0, # 通常不需要，除非有明显噪点
        'gamma': 1.2,                      # 尝试1.0 (不调整), 1.2 (略微压暗亮区，可能增强对比), 0.8 (提亮暗区)
                                           # DRIVE图像通常质量较好，gamma调整可能不明显或非必需
        'use_morphological_enhancement': True, # 启用形态学增强
        'morph_kernel_size': 15,           # **关键参数** 尝试 11, 15, 21。应略大于最粗血管的直径
                                           # DRIVE血管相对较细，15可能合适。如果过大，会丢失细血管。
        'morph_op_iterations': 1           # 通常1次迭代足够
    }
    
    # 如果是病灶非常严重的图像集，可能需要更大的 background_kernel_size (如201-301)
    # 同时 morph_kernel_size 可能也需要相应调整

    process_directory(
        input_images_dir,
        output_processed_dir,
        mask_dir=mask_files_dir,
        **drive_preprocessor_config
    )
    print("所有图像处理完成。")

    # --- 测试一张特定的图像 ---
    test_image_filename = "21_training.tif" # DRIVE数据集的第一张图，或者你关注的图
    test_mask_filename_stem = "21_training_mask" # DRIVE的mask通常是 .gif

    # 灵活查找mask文件
    test_problem_image_path = os.path.join(input_images_dir, test_image_filename)
    test_problem_mask_path = None
    if mask_files_dir:
        mask_extensions = ['.gif', '.png', '.tif'] # 可能的mask扩展名
        for ext in mask_extensions:
            candidate_mask = os.path.join(mask_files_dir, f"{test_mask_filename_stem}{ext}")
            if os.path.exists(candidate_mask):
                test_problem_mask_path = candidate_mask
                break
    
    if os.path.exists(test_problem_image_path):
        print(f"\n正在测试单张图像: {test_problem_image_path}")
        if test_problem_mask_path:
            print(f"使用Mask: {test_problem_mask_path}")
        else:
            print("未找到对应Mask或未使用Mask进行测试。")
            
        test_preprocessor_instance = ImagePreprocessor(**drive_preprocessor_config)
        test_preprocessor_instance.display_images(test_problem_image_path, test_problem_mask_path)
    else:
        print(f"\n测试图像文件不存在。请检查路径: {test_problem_image_path}")

