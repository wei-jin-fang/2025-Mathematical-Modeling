import os
import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import label
import pandas as pd


def sine_func(x, R, P, beta, C):
    return R * np.sin(2 * np.pi * x / P + beta) + C


def fit_sine_to_crack(points):
    if len(points) < 4:  # 需要至少4个点来拟合4个参数
        return None
    x_data = points[:, 0]
    y_data = points[:, 1]
    # 初始猜测
    C_guess = np.mean(y_data)
    R_guess = (np.max(y_data) - np.min(y_data)) / 2
    P_guess = np.max(x_data) - np.min(x_data) + 1  # 近似宽度
    beta_guess = 0.0
    p0 = [R_guess, P_guess, beta_guess, C_guess]
    try:
        popt, pcov = curve_fit(sine_func, x_data, y_data, p0=p0, maxfev=10000)
        return popt
    except Exception as e:
        print(f"拟合失败: {e}")
        return None


def process_binary_mask(file_path):
    img = cv2.imread(file_path, 0)
    if img is None:
        print(f"无法加载图像：{file_path}")
        return []
    crack = (img == 0)
    labeled, num_features = label(crack)
    results = []
    for i in range(1, num_features + 1):
        y, x = np.where(labeled == i)
        if len(x) < 4:
            continue
        points = np.column_stack((x, y))  # (x, y)
        params = fit_sine_to_crack(points)
        if params is not None:
            R, P, beta, C = params
            results.append((i, R, P, beta, C))
    return results


# 主程序
if __name__ == '__main__':
    output_dir = '../q1/output/predict'
    binary_files = [f for f in os.listdir(output_dir) if f.startswith('binary_mask_')]
    data = []
    for file in sorted(binary_files):
        # 提取图像编号，假设 filename = binary_mask_xxx.jpg
        image_id = file.replace('binary_mask_', '').rsplit('.', 1)[0]
        path = os.path.join(output_dir, file)
        res = process_binary_mask(path)
        for crack_id, R, P, beta, C in res:
            data.append((image_id, crack_id, R, P, beta, C))

    # 创建 DataFrame
    columns = ['图像编号', '裂隙编号', '振幅R(mm)', '周期P (mm)', '相位β (rad)', '中心线位置C (mm)']
    df = pd.DataFrame(data, columns=columns)

    # 打印表格
    print("表1 “正弦状”裂隙的定量分析建模结果汇总")
    print(df.to_string(index=False))