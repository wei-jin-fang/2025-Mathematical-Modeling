import cv2
import os
import numpy as np

# 设置输入和输出文件夹路径
input_folder = "./data/image/train"  # 替换为你的图像文件夹路径
output_folder = "./predict/canny"  # 替换为保存边缘检测结果的文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 支持的图像格式
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

# 遍历文件夹中的所有图像
for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_extensions):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像: {filename}")
            continue

        # 将RGB图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 可选：对图像进行模糊处理以减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 应用Canny边缘检测，使用更宽松的阈值
        edges = cv2.Canny(blurred, threshold1=100, threshold2=150)  # 降低阈值以检测更多边缘

        # 保存边缘检测结果
        output_path = os.path.join(output_folder, f"edge_{filename}")
        cv2.imwrite(output_path, edges)
        print(f"已处理并保存: {output_path}")

print("边缘检测完成！")