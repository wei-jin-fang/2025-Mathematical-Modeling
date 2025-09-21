import cv2
import numpy as np
import os

# 全局变量
drawing = False  # 是否正在绘制
brush_size = 5  # 初始刷子大小
mask = None  # 当前mask
display_scale = 1.0  # 显示缩放比例


# 鼠标回调函数
def draw_mask(event, x, y, flags, param):
    global drawing, brush_size, mask, display_scale
    # 将显示坐标转换回原图坐标
    x_orig = int(x / display_scale)
    y_orig = int(y / display_scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x_orig, y_orig), brush_size, 255, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x_orig, y_orig), brush_size, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# 主函数：生成mask，适配小屏幕
def generate_masks(image_dir, mask_dir):
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法加载图像：{img_path}")
            continue

        # 原图尺寸
        height, width = img.shape[:2]
        global mask
        mask = np.zeros((height, width), dtype=np.uint8)  # 原分辨率mask

        # 计算显示缩放比例，最大800x600
        max_display_size = (800, 600)
        global display_scale
        display_scale = min(max_display_size[0] / width, max_display_size[1] / height, 1.0)
        display_size = (int(width * display_scale), int(height * display_scale))

        window_name = f"绘制Mask: {img_file} (s:保存, c:清除, +/-:调整大小, esc:退出)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_size[0], display_size[1])
        cv2.setMouseCallback(window_name, draw_mask)
        brush_size=None
        while True:
            # 调整显示图像
            display_img = cv2.resize(img, display_size, interpolation=cv2.INTER_AREA)
            display_mask = cv2.resize(mask, display_size, interpolation=cv2.INTER_NEAREST)
            # 叠加mask（红色）
            overlay = display_img.copy()
            overlay[display_mask > 0] = [0, 0, 255]
            display = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)
            # 显示刷子大小
            cv2.putText(display, f"刷子: {brush_size}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 保存
                mask_file = img_file.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
                mask_path = os.path.join(mask_dir, mask_file)
                cv2.imwrite(mask_path, mask)  # 保存原分辨率mask
                print(f"已保存mask至：{mask_path}")
                break
            elif key == ord('c'):  # 清除
                mask = np.zeros((height, width), dtype=np.uint8)
            elif key == ord('+'):  # 增大刷子
                brush_size = min(brush_size + 1, 50)
            elif key == ord('-'):  # 减小刷子
                brush_size = max(brush_size - 1, 1)
            elif key == 27:  # esc退出
                print(f"跳过图像：{img_file}")
                break

        cv2.destroyAllWindows()


# 示例使用：替换为你的路径
if __name__ == "__main__":
    image_dir = './data/image/'  # 与训练代码相同
    mask_dir = './data/mask/'  # 与训练代码相同
    generate_masks(image_dir, mask_dir)