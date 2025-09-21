import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
# UNet模型定义
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# 自定义推理数据集（仅加载图像）
class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法加载图像：{img_path}")
            return None



        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, self.images[idx]  # 返回图像和文件名

# 数据变换（与训练时一致）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 推理函数（仅保存原图和预测对比图）
def infer_images(model, loader, device, output_dir='./predict/'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    with torch.no_grad():
        for i, (images, filenames) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # 二值化

            for j, (pred, filename) in enumerate(zip(preds, filenames)):
                # 反标准化图像以正确显示
                image = images[j].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image * std + mean) * 255.0  # 反标准化并缩放到[0, 255]
                image = image.astype(np.uint8)

                # 检查预测掩码方向并修正
                pred_np = pred.cpu().squeeze().numpy()


                # 可视化（原图 + 预测掩码）
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 2, 1)
                plt.title('Input Image')
                plt.imshow(image)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title('Predicted Binary Mask')
                plt.imshow(pred_np, cmap='gray')
                plt.axis('off')

                # 保存对比图
                output_path = os.path.join(output_dir, f'comparison_{filename}')
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                print(f"对比图像已保存至：{output_path}")
                plt.close()

# 主程序
if __name__ == '__main__':
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    checkpoint_path = './checkpoints/checkpoint_epoch_500.pth'  # 假设使用最后一轮的模型
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件 {checkpoint_path} 不存在！")
        exit()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模型权重：{checkpoint_path}")

    # 创建推理数据集和DataLoader
    image_dir = './data/image/'  # 指定读取的文件夹
    inference_dataset = InferenceDataset(image_dir, transform=transform)
    inference_loader = DataLoader(inference_dataset, batch_size=2, shuffle=False)  # batch_size可调整

    # 推理并保存对比图
    infer_images(model, inference_loader, device)