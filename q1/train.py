import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# 自定义数据集
class FractureDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_file = os.path.splitext(self.images[idx])[0] + '_mask_mask.png'
        mask_path = os.path.join(self.mask_dir, mask_file)

        # 加载图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法加载图像：{img_path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法加载mask：{mask_path}")
            return None

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


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


# 数据变换（调整为244x1350）
transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((244, 1350)),  # 保持原始尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize((244, 1350), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

# 数据路径
image_dir = './data/image/'
mask_dir = './data/mask/'

# 创建数据集
full_dataset = FractureDataset(image_dir, mask_dir, transform=transform, mask_transform=mask_transform)

# 使用同一数据集作为训练、验证、测试集
train_dataset = full_dataset
val_dataset = full_dataset
test_dataset = full_dataset

# DataLoader
batch_size = 10  # 由于图像尺寸较大，减小batch_size以适应显存
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失、优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # 用于二值化输出
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


# 验证函数
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
    return running_loss / len(loader)


# 测试函数（包括输出二值化图示例）
def test_model(model, loader, device, num_examples=3):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if i >= num_examples:
                break
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # 二值化

            # 反标准化图像以正确显示
            image = images[0].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image * std + mean) * 255.0  # 反标准化并缩放到[0, 255]
            image = image.astype(np.uint8)

            # 可视化
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title('Input Image')
            plt.imshow(image)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth Mask')
            plt.imshow(masks[0].cpu().squeeze(), cmap='gray')  # 去除通道维度
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.title('Predicted Binary Mask')
            plt.imshow(preds[0].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            plt.show()


# 创建保存模型的目录
os.makedirs('./checkpoints', exist_ok=True)


def train():
    # 训练循环
    num_epochs = 1000
    save_interval = 5  # 每5轮保存一次
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 每5轮保存模型
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'./checkpoints/checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'模型已保存至 {checkpoint_path}')


if __name__ == '__main__':
    train()

    '''
    模型已保存至 ./checkpoints/checkpoint_epoch_155.pth
Epoch 156/1000, Train Loss: 0.0438, Val Loss: 0.0470
Epoch 157/1000, Train Loss: 0.0432, Val Loss: 0.0447
Epoch 158/1000, Train Loss: 0.0427, Val Loss: 0.0443
Epoch 159/1000, Train Loss: 0.0423, Val Loss: 0.0442
Epoch 160/1000, Train Loss: 0.0417, Val Loss: 0.0431
模型已保存至 ./checkpoints/checkpoint_epoch_160.pth
Epoch 161/1000, Train Loss: 0.0412, Val Loss: 0.0446
Epoch 162/1000, Train Loss: 0.0411, Val Loss: 0.0435
Epoch 163/1000, Train Loss: 0.0409, Val Loss: 0.0445
Epoch 164/1000, Train Loss: 0.0426, Val Loss: 0.0444
Epoch 165/1000, Train Loss: 0.0406, Val Loss: 0.0594
模型已保存至 ./checkpoints/checkpoint_epoch_165.pth
Epoch 166/1000, Train Loss: 0.0403, Val Loss: 0.0553
Epoch 167/1000, Train Loss: 0.0409, Val Loss: 0.0589
Epoch 168/1000, Train Loss: 0.0398, Val Loss: 0.0577
Epoch 169/1000, Train Loss: 0.0400, Val Loss: 0.0443
Epoch 170/1000, Train Loss: 0.0388, Val Loss: 0.0410
模型已保存至 ./checkpoints/checkpoint_epoch_170.pth
Epoch 171/1000, Train Loss: 0.0389, Val Loss: 0.0398
Epoch 172/1000, Train Loss: 0.0381, Val Loss: 0.0393
Epoch 173/1000, Train Loss: 0.0375, Val Loss: 0.0389
Epoch 174/1000, Train Loss: 0.0370, Val Loss: 0.0382
Epoch 175/1000, Train Loss: 0.0364, Val Loss: 0.0381
模型已保存至 ./checkpoints/checkpoint_epoch_175.pth
Epoch 176/1000, Train Loss: 0.0361, Val Loss: 0.0375
Epoch 177/1000, Train Loss: 0.0357, Val Loss: 0.0378
Epoch 178/1000, Train Loss: 0.0353, Val Loss: 0.0383
Epoch 179/1000, Train Loss: 0.0349, Val Loss: 0.0375
Epoch 180/1000, Train Loss: 0.0345, Val Loss: 0.0372
模型已保存至 ./checkpoints/checkpoint_epoch_180.pth
Epoch 181/1000, Train Loss: 0.0341, Val Loss: 0.0364
Epoch 182/1000, Train Loss: 0.0336, Val Loss: 0.0371
Epoch 183/1000, Train Loss: 0.0335, Val Loss: 0.0344
Epoch 184/1000, Train Loss: 0.0330, Val Loss: 0.0343
Epoch 185/1000, Train Loss: 0.0327, Val Loss: 0.0338
模型已保存至 ./checkpoints/checkpoint_epoch_185.pth
Epoch 186/1000, Train Loss: 0.0323, Val Loss: 0.0347
Epoch 187/1000, Train Loss: 0.0324, Val Loss: 0.0339
Epoch 188/1000, Train Loss: 0.0323, Val Loss: 0.0328
Epoch 189/1000, Train Loss: 0.0320, Val Loss: 0.0318
Epoch 190/1000, Train Loss: 0.0309, Val Loss: 0.0328
模型已保存至 ./checkpoints/checkpoint_epoch_190.pth
Epoch 191/1000, Train Loss: 0.0318, Val Loss: 0.0340
Epoch 192/1000, Train Loss: 0.0324, Val Loss: 0.0354
Epoch 193/1000, Train Loss: 0.0322, Val Loss: 0.0320
Epoch 194/1000, Train Loss: 0.0307, Val Loss: 0.0316
Epoch 195/1000, Train Loss: 0.0305, Val Loss: 0.0307
模型已保存至 ./checkpoints/checkpoint_epoch_195.pth
Epoch 196/1000, Train Loss: 0.0301, Val Loss: 0.0301
Epoch 197/1000, Train Loss: 0.0295, Val Loss: 0.0305
Epoch 198/1000, Train Loss: 0.0291, Val Loss: 0.0304
Epoch 199/1000, Train Loss: 0.0288, Val Loss: 0.0299
Epoch 200/1000, Train Loss: 0.0284, Val Loss: 0.0292
模型已保存至 ./checkpoints/checkpoint_epoch_200.pth
Epoch 201/1000, Train Loss: 0.0279, Val Loss: 0.0294
Epoch 202/1000, Train Loss: 0.0273, Val Loss: 0.0376
Epoch 203/1000, Train Loss: 0.0273, Val Loss: 0.0270
Epoch 204/1000, Train Loss: 0.0270, Val Loss: 0.0280
Epoch 205/1000, Train Loss: 0.0265, Val Loss: 0.0274
模型已保存至 ./checkpoints/checkpoint_epoch_205.pth
Epoch 206/1000, Train Loss: 0.0262, Val Loss: 0.0287
Epoch 207/1000, Train Loss: 0.0261, Val Loss: 0.0275
Epoch 208/1000, Train Loss: 0.0254, Val Loss: 0.0300
Epoch 209/1000, Train Loss: 0.0259, Val Loss: 0.0271
Epoch 210/1000, Train Loss: 0.0259, Val Loss: 0.0304
模型已保存至 ./checkpoints/checkpoint_epoch_210.pth
Epoch 211/1000, Train Loss: 0.0272, Val Loss: 0.0286
Epoch 212/1000, Train Loss: 0.0262, Val Loss: 0.0285
Epoch 213/1000, Train Loss: 0.0255, Val Loss: 0.0252
Epoch 214/1000, Train Loss: 0.0256, Val Loss: 0.0268
Epoch 215/1000, Train Loss: 0.0247, Val Loss: 0.0284
模型已保存至 ./checkpoints/checkpoint_epoch_215.pth
Epoch 216/1000, Train Loss: 0.0247, Val Loss: 0.0252
Epoch 217/1000, Train Loss: 0.0244, Val Loss: 0.0247
Epoch 218/1000, Train Loss: 0.0235, Val Loss: 0.0247
Epoch 219/1000, Train Loss: 0.0234, Val Loss: 0.0259
Epoch 220/1000, Train Loss: 0.0229, Val Loss: 0.0255
模型已保存至 ./checkpoints/checkpoint_epoch_220.pth
Epoch 221/1000, Train Loss: 0.0226, Val Loss: 0.0241
Epoch 222/1000, Train Loss: 0.0222, Val Loss: 0.0221
Epoch 223/1000, Train Loss: 0.0219, Val Loss: 0.0223
Epoch 224/1000, Train Loss: 0.0220, Val Loss: 0.0220
Epoch 225/1000, Train Loss: 0.0219, Val Loss: 0.0225
模型已保存至 ./checkpoints/checkpoint_epoch_225.pth
Epoch 226/1000, Train Loss: 0.0212, Val Loss: 0.0226
Epoch 227/1000, Train Loss: 0.0214, Val Loss: 0.0214
Epoch 228/1000, Train Loss: 0.0209, Val Loss: 0.0218
Epoch 229/1000, Train Loss: 0.0208, Val Loss: 0.0221
Epoch 230/1000, Train Loss: 0.0203, Val Loss: 0.0218
模型已保存至 ./checkpoints/checkpoint_epoch_230.pth
Epoch 231/1000, Train Loss: 0.0202, Val Loss: 0.0211
Epoch 232/1000, Train Loss: 0.0200, Val Loss: 0.0204
Epoch 233/1000, Train Loss: 0.0198, Val Loss: 0.0199
Epoch 234/1000, Train Loss: 0.0195, Val Loss: 0.0202
Epoch 235/1000, Train Loss: 0.0193, Val Loss: 0.0201
模型已保存至 ./checkpoints/checkpoint_epoch_235.pth
Epoch 236/1000, Train Loss: 0.0194, Val Loss: 0.0205
Epoch 237/1000, Train Loss: 0.0195, Val Loss: 0.0226
Epoch 238/1000, Train Loss: 0.0203, Val Loss: 0.0224
Epoch 239/1000, Train Loss: 0.0212, Val Loss: 0.0238
Epoch 240/1000, Train Loss: 0.0202, Val Loss: 0.0219
模型已保存至 ./checkpoints/checkpoint_epoch_240.pth
Epoch 241/1000, Train Loss: 0.0193, Val Loss: 0.0201
Epoch 242/1000, Train Loss: 0.0197, Val Loss: 0.0188
Epoch 243/1000, Train Loss: 0.0186, Val Loss: 0.0189
Epoch 244/1000, Train Loss: 0.0192, Val Loss: 0.0210
Epoch 245/1000, Train Loss: 0.0187, Val Loss: 0.0194
模型已保存至 ./checkpoints/checkpoint_epoch_245.pth
Epoch 246/1000, Train Loss: 0.0182, Val Loss: 0.0204
Epoch 247/1000, Train Loss: 0.0183, Val Loss: 0.0181
Epoch 248/1000, Train Loss: 0.0179, Val Loss: 0.0180
Epoch 249/1000, Train Loss: 0.0178, Val Loss: 0.0177
Epoch 250/1000, Train Loss: 0.0176, Val Loss: 0.0182
模型已保存至 ./checkpoints/checkpoint_epoch_250.pth
Epoch 251/1000, Train Loss: 0.0174, Val Loss: 0.0172
Epoch 252/1000, Train Loss: 0.0173, Val Loss: 0.0170
Epoch 253/1000, Train Loss: 0.0170, Val Loss: 0.0170
Epoch 254/1000, Train Loss: 0.0170, Val Loss: 0.0168
Epoch 255/1000, Train Loss: 0.0166, Val Loss: 0.0169
模型已保存至 ./checkpoints/checkpoint_epoch_255.pth
Epoch 256/1000, Train Loss: 0.0166, Val Loss: 0.0176
Epoch 257/1000, Train Loss: 0.0163, Val Loss: 0.0177
Epoch 258/1000, Train Loss: 0.0163, Val Loss: 0.0167
Epoch 259/1000, Train Loss: 0.0161, Val Loss: 0.0164
Epoch 260/1000, Train Loss: 0.0159, Val Loss: 0.0162
模型已保存至 ./checkpoints/checkpoint_epoch_260.pth
Epoch 261/1000, Train Loss: 0.0157, Val Loss: 0.0167
Epoch 262/1000, Train Loss: 0.0156, Val Loss: 0.0167
Epoch 263/1000, Train Loss: 0.0156, Val Loss: 0.0163
Epoch 264/1000, Train Loss: 0.0155, Val Loss: 0.0161
Epoch 265/1000, Train Loss: 0.0155, Val Loss: 0.0171
模型已保存至 ./checkpoints/checkpoint_epoch_265.pth
Epoch 266/1000, Train Loss: 0.0154, Val Loss: 0.0157
Epoch 267/1000, Train Loss: 0.0152, Val Loss: 0.0154
Epoch 268/1000, Train Loss: 0.0149, Val Loss: 0.0152
Epoch 269/1000, Train Loss: 0.0148, Val Loss: 0.0151
Epoch 270/1000, Train Loss: 0.0147, Val Loss: 0.0150
模型已保存至 ./checkpoints/checkpoint_epoch_270.pth
Epoch 271/1000, Train Loss: 0.0147, Val Loss: 0.0148
Epoch 272/1000, Train Loss: 0.0145, Val Loss: 0.0147
Epoch 273/1000, Train Loss: 0.0144, Val Loss: 0.0144
Epoch 274/1000, Train Loss: 0.0143, Val Loss: 0.0143
Epoch 275/1000, Train Loss: 0.0142, Val Loss: 0.0140
模型已保存至 ./checkpoints/checkpoint_epoch_275.pth
Epoch 276/1000, Train Loss: 0.0140, Val Loss: 0.0139
Epoch 277/1000, Train Loss: 0.0139, Val Loss: 0.0138
Epoch 278/1000, Train Loss: 0.0138, Val Loss: 0.0138
Epoch 279/1000, Train Loss: 0.0138, Val Loss: 0.0138
Epoch 280/1000, Train Loss: 0.0137, Val Loss: 0.0136
模型已保存至 ./checkpoints/checkpoint_epoch_280.pth
Epoch 281/1000, Train Loss: 0.0136, Val Loss: 0.0138
Epoch 282/1000, Train Loss: 0.0136, Val Loss: 0.0135
Epoch 283/1000, Train Loss: 0.0136, Val Loss: 0.0140
Epoch 284/1000, Train Loss: 0.0136, Val Loss: 0.0135
Epoch 285/1000, Train Loss: 0.0136, Val Loss: 0.0139
模型已保存至 ./checkpoints/checkpoint_epoch_285.pth
Epoch 286/1000, Train Loss: 0.0136, Val Loss: 0.0132
Epoch 287/1000, Train Loss: 0.0134, Val Loss: 0.0131
Epoch 288/1000, Train Loss: 0.0130, Val Loss: 0.0130
Epoch 289/1000, Train Loss: 0.0128, Val Loss: 0.0129
Epoch 290/1000, Train Loss: 0.0129, Val Loss: 0.0131
模型已保存至 ./checkpoints/checkpoint_epoch_290.pth
Epoch 291/1000, Train Loss: 0.0129, Val Loss: 0.0127
Epoch 292/1000, Train Loss: 0.0127, Val Loss: 0.0127
Epoch 293/1000, Train Loss: 0.0125, Val Loss: 0.0126
Epoch 294/1000, Train Loss: 0.0125, Val Loss: 0.0125
Epoch 295/1000, Train Loss: 0.0125, Val Loss: 0.0127
模型已保存至 ./checkpoints/checkpoint_epoch_295.pth
Epoch 296/1000, Train Loss: 0.0124, Val Loss: 0.0124
Epoch 297/1000, Train Loss: 0.0123, Val Loss: 0.0123
Epoch 298/1000, Train Loss: 0.0121, Val Loss: 0.0122
Epoch 299/1000, Train Loss: 0.0121, Val Loss: 0.0123
Epoch 300/1000, Train Loss: 0.0121, Val Loss: 0.0122
模型已保存至 ./checkpoints/checkpoint_epoch_300.pth
Epoch 301/1000, Train Loss: 0.0120, Val Loss: 0.0126
Epoch 302/1000, Train Loss: 0.0121, Val Loss: 0.0125
Epoch 303/1000, Train Loss: 0.0122, Val Loss: 0.0148
Epoch 304/1000, Train Loss: 0.0129, Val Loss: 0.0149
Epoch 305/1000, Train Loss: 0.0125, Val Loss: 0.0130
模型已保存至 ./checkpoints/checkpoint_epoch_305.pth
Epoch 306/1000, Train Loss: 0.0120, Val Loss: 0.0121
Epoch 307/1000, Train Loss: 0.0117, Val Loss: 0.0132
Epoch 308/1000, Train Loss: 0.0120, Val Loss: 0.0117
Epoch 309/1000, Train Loss: 0.0116, Val Loss: 0.0117
Epoch 310/1000, Train Loss: 0.0116, Val Loss: 0.0118
模型已保存至 ./checkpoints/checkpoint_epoch_310.pth
Epoch 311/1000, Train Loss: 0.0116, Val Loss: 0.0117
Epoch 312/1000, Train Loss: 0.0113, Val Loss: 0.0114
Epoch 313/1000, Train Loss: 0.0114, Val Loss: 0.0113
Epoch 314/1000, Train Loss: 0.0112, Val Loss: 0.0116
Epoch 315/1000, Train Loss: 0.0112, Val Loss: 0.0111
模型已保存至 ./checkpoints/checkpoint_epoch_315.pth
Epoch 316/1000, Train Loss: 0.0111, Val Loss: 0.0110
Epoch 317/1000, Train Loss: 0.0110, Val Loss: 0.0110
Epoch 318/1000, Train Loss: 0.0109, Val Loss: 0.0112
Epoch 319/1000, Train Loss: 0.0109, Val Loss: 0.0108
Epoch 320/1000, Train Loss: 0.0108, Val Loss: 0.0110
模型已保存至 ./checkpoints/checkpoint_epoch_320.pth
Epoch 321/1000, Train Loss: 0.0107, Val Loss: 0.0108
Epoch 322/1000, Train Loss: 0.0107, Val Loss: 0.0111
Epoch 323/1000, Train Loss: 0.0106, Val Loss: 0.0106
Epoch 324/1000, Train Loss: 0.0106, Val Loss: 0.0111
Epoch 325/1000, Train Loss: 0.0106, Val Loss: 0.0106
模型已保存至 ./checkpoints/checkpoint_epoch_325.pth
Epoch 326/1000, Train Loss: 0.0107, Val Loss: 0.0115
Epoch 327/1000, Train Loss: 0.0110, Val Loss: 0.0112
Epoch 328/1000, Train Loss: 0.0114, Val Loss: 0.0120
Epoch 329/1000, Train Loss: 0.0116, Val Loss: 0.0107
Epoch 330/1000, Train Loss: 0.0107, Val Loss: 0.0102
模型已保存至 ./checkpoints/checkpoint_epoch_330.pth
Epoch 331/1000, Train Loss: 0.0102, Val Loss: 0.0108
Epoch 332/1000, Train Loss: 0.0106, Val Loss: 0.0110
Epoch 333/1000, Train Loss: 0.0105, Val Loss: 0.0100
Epoch 334/1000, Train Loss: 0.0100, Val Loss: 0.0102
Epoch 335/1000, Train Loss: 0.0102, Val Loss: 0.0104
模型已保存至 ./checkpoints/checkpoint_epoch_335.pth
Epoch 336/1000, Train Loss: 0.0102, Val Loss: 0.0101
Epoch 337/1000, Train Loss: 0.0099, Val Loss: 0.0100
Epoch 338/1000, Train Loss: 0.0100, Val Loss: 0.0099
Epoch 339/1000, Train Loss: 0.0099, Val Loss: 0.0099
Epoch 340/1000, Train Loss: 0.0097, Val Loss: 0.0099
模型已保存至 ./checkpoints/checkpoint_epoch_340.pth
Epoch 341/1000, Train Loss: 0.0098, Val Loss: 0.0097
Epoch 342/1000, Train Loss: 0.0096, Val Loss: 0.0098
Epoch 343/1000, Train Loss: 0.0095, Val Loss: 0.0098
Epoch 344/1000, Train Loss: 0.0096, Val Loss: 0.0098
Epoch 345/1000, Train Loss: 0.0094, Val Loss: 0.0097
模型已保存至 ./checkpoints/checkpoint_epoch_345.pth
Epoch 346/1000, Train Loss: 0.0094, Val Loss: 0.0097
Epoch 347/1000, Train Loss: 0.0094, Val Loss: 0.0096
Epoch 348/1000, Train Loss: 0.0093, Val Loss: 0.0096
Epoch 349/1000, Train Loss: 0.0092, Val Loss: 0.0095
Epoch 350/1000, Train Loss: 0.0092, Val Loss: 0.0094
模型已保存至 ./checkpoints/checkpoint_epoch_350.pth
Epoch 351/1000, Train Loss: 0.0091, Val Loss: 0.0094
Epoch 352/1000, Train Loss: 0.0091, Val Loss: 0.0094
Epoch 353/1000, Train Loss: 0.0090, Val Loss: 0.0092
Epoch 354/1000, Train Loss: 0.0090, Val Loss: 0.0091
Epoch 355/1000, Train Loss: 0.0089, Val Loss: 0.0091
模型已保存至 ./checkpoints/checkpoint_epoch_355.pth
Epoch 356/1000, Train Loss: 0.0089, Val Loss: 0.0091
Epoch 357/1000, Train Loss: 0.0088, Val Loss: 0.0090
Epoch 358/1000, Train Loss: 0.0088, Val Loss: 0.0088
Epoch 359/1000, Train Loss: 0.0087, Val Loss: 0.0089
Epoch 360/1000, Train Loss: 0.0087, Val Loss: 0.0087
模型已保存至 ./checkpoints/checkpoint_epoch_360.pth
Epoch 361/1000, Train Loss: 0.0087, Val Loss: 0.0088
Epoch 362/1000, Train Loss: 0.0087, Val Loss: 0.0086
Epoch 363/1000, Train Loss: 0.0087, Val Loss: 0.0091
Epoch 364/1000, Train Loss: 0.0089, Val Loss: 0.0085
Epoch 365/1000, Train Loss: 0.0086, Val Loss: 0.0084
模型已保存至 ./checkpoints/checkpoint_epoch_365.pth
Epoch 366/1000, Train Loss: 0.0085, Val Loss: 0.0083
Epoch 367/1000, Train Loss: 0.0084, Val Loss: 0.0083
Epoch 368/1000, Train Loss: 0.0084, Val Loss: 0.0084
Epoch 369/1000, Train Loss: 0.0084, Val Loss: 0.0084
Epoch 370/1000, Train Loss: 0.0083, Val Loss: 0.0082
模型已保存至 ./checkpoints/checkpoint_epoch_370.pth
Epoch 371/1000, Train Loss: 0.0082, Val Loss: 0.0083
Epoch 372/1000, Train Loss: 0.0083, Val Loss: 0.0082
Epoch 373/1000, Train Loss: 0.0082, Val Loss: 0.0081
Epoch 374/1000, Train Loss: 0.0081, Val Loss: 0.0081
Epoch 375/1000, Train Loss: 0.0081, Val Loss: 0.0081
模型已保存至 ./checkpoints/checkpoint_epoch_375.pth
Epoch 376/1000, Train Loss: 0.0080, Val Loss: 0.0081
Epoch 377/1000, Train Loss: 0.0080, Val Loss: 0.0080
Epoch 378/1000, Train Loss: 0.0079, Val Loss: 0.0081
Epoch 379/1000, Train Loss: 0.0079, Val Loss: 0.0080
Epoch 380/1000, Train Loss: 0.0079, Val Loss: 0.0081
模型已保存至 ./checkpoints/checkpoint_epoch_380.pth
Epoch 381/1000, Train Loss: 0.0079, Val Loss: 0.0080
Epoch 382/1000, Train Loss: 0.0080, Val Loss: 0.0085
Epoch 383/1000, Train Loss: 0.0082, Val Loss: 0.0082
Epoch 384/1000, Train Loss: 0.0085, Val Loss: 0.0090
Epoch 385/1000, Train Loss: 0.0088, Val Loss: 0.0082
模型已保存至 ./checkpoints/checkpoint_epoch_385.pth
Epoch 386/1000, Train Loss: 0.0084, Val Loss: 0.0078
Epoch 387/1000, Train Loss: 0.0081, Val Loss: 0.0088
Epoch 388/1000, Train Loss: 0.0084, Val Loss: 0.0078
Epoch 389/1000, Train Loss: 0.0079, Val Loss: 0.0078
Epoch 390/1000, Train Loss: 0.0082, Val Loss: 0.0080
模型已保存至 ./checkpoints/checkpoint_epoch_390.pth
Epoch 391/1000, Train Loss: 0.0082, Val Loss: 0.0076
Epoch 392/1000, Train Loss: 0.0079, Val Loss: 0.0074
Epoch 393/1000, Train Loss: 0.0079, Val Loss: 0.0083
Epoch 394/1000, Train Loss: 0.0079, Val Loss: 0.0075
Epoch 395/1000, Train Loss: 0.0077, Val Loss: 0.0076
模型已保存至 ./checkpoints/checkpoint_epoch_395.pth
Epoch 396/1000, Train Loss: 0.0077, Val Loss: 0.0077
Epoch 397/1000, Train Loss: 0.0076, Val Loss: 0.0075
Epoch 398/1000, Train Loss: 0.0075, Val Loss: 0.0075
Epoch 399/1000, Train Loss: 0.0075, Val Loss: 0.0075
Epoch 400/1000, Train Loss: 0.0074, Val Loss: 0.0073
模型已保存至 ./checkpoints/checkpoint_epoch_400.pth
Epoch 401/1000, Train Loss: 0.0074, Val Loss: 0.0075
Epoch 402/1000, Train Loss: 0.0073, Val Loss: 0.0075
Epoch 403/1000, Train Loss: 0.0073, Val Loss: 0.0073
Epoch 404/1000, Train Loss: 0.0072, Val Loss: 0.0073
Epoch 405/1000, Train Loss: 0.0072, Val Loss: 0.0072
模型已保存至 ./checkpoints/checkpoint_epoch_405.pth
Epoch 406/1000, Train Loss: 0.0071, Val Loss: 0.0073
Epoch 407/1000, Train Loss: 0.0071, Val Loss: 0.0073
Epoch 408/1000, Train Loss: 0.0071, Val Loss: 0.0072
Epoch 409/1000, Train Loss: 0.0070, Val Loss: 0.0072
Epoch 410/1000, Train Loss: 0.0069, Val Loss: 0.0072
模型已保存至 ./checkpoints/checkpoint_epoch_410.pth
Epoch 411/1000, Train Loss: 0.0069, Val Loss: 0.0071
Epoch 412/1000, Train Loss: 0.0069, Val Loss: 0.0071
Epoch 413/1000, Train Loss: 0.0069, Val Loss: 0.0071
Epoch 414/1000, Train Loss: 0.0068, Val Loss: 0.0071
Epoch 415/1000, Train Loss: 0.0068, Val Loss: 0.0070
模型已保存至 ./checkpoints/checkpoint_epoch_415.pth
Epoch 416/1000, Train Loss: 0.0067, Val Loss: 0.0070
Epoch 417/1000, Train Loss: 0.0067, Val Loss: 0.0069
Epoch 418/1000, Train Loss: 0.0067, Val Loss: 0.0069
Epoch 419/1000, Train Loss: 0.0066, Val Loss: 0.0069
Epoch 420/1000, Train Loss: 0.0066, Val Loss: 0.0068
模型已保存至 ./checkpoints/checkpoint_epoch_420.pth
Epoch 421/1000, Train Loss: 0.0066, Val Loss: 0.0068
Epoch 422/1000, Train Loss: 0.0065, Val Loss: 0.0067
Epoch 423/1000, Train Loss: 0.0065, Val Loss: 0.0067
Epoch 424/1000, Train Loss: 0.0065, Val Loss: 0.0066
Epoch 425/1000, Train Loss: 0.0064, Val Loss: 0.0066
模型已保存至 ./checkpoints/checkpoint_epoch_425.pth
Epoch 426/1000, Train Loss: 0.0064, Val Loss: 0.0066
Epoch 427/1000, Train Loss: 0.0064, Val Loss: 0.0065
Epoch 428/1000, Train Loss: 0.0064, Val Loss: 0.0066
Epoch 429/1000, Train Loss: 0.0064, Val Loss: 0.0066
Epoch 430/1000, Train Loss: 0.0065, Val Loss: 0.0074
模型已保存至 ./checkpoints/checkpoint_epoch_430.pth
Epoch 431/1000, Train Loss: 0.0070, Val Loss: 0.0076
Epoch 432/1000, Train Loss: 0.0072, Val Loss: 0.0076
Epoch 433/1000, Train Loss: 0.0075, Val Loss: 0.0069
Epoch 434/1000, Train Loss: 0.0067, Val Loss: 0.0075
Epoch 435/1000, Train Loss: 0.0068, Val Loss: 0.0068
模型已保存至 ./checkpoints/checkpoint_epoch_435.pth
Epoch 436/1000, Train Loss: 0.0070, Val Loss: 0.0067
Epoch 437/1000, Train Loss: 0.0064, Val Loss: 0.0097
Epoch 438/1000, Train Loss: 0.0069, Val Loss: 0.0069
Epoch 439/1000, Train Loss: 0.0065, Val Loss: 0.0080
Epoch 440/1000, Train Loss: 0.0067, Val Loss: 0.0072
模型已保存至 ./checkpoints/checkpoint_epoch_440.pth
Epoch 441/1000, Train Loss: 0.0064, Val Loss: 0.0081
Epoch 442/1000, Train Loss: 0.0066, Val Loss: 0.0063
Epoch 443/1000, Train Loss: 0.0064, Val Loss: 0.0066
Epoch 444/1000, Train Loss: 0.0064, Val Loss: 0.0063
Epoch 445/1000, Train Loss: 0.0063, Val Loss: 0.0064
模型已保存至 ./checkpoints/checkpoint_epoch_445.pth
Epoch 446/1000, Train Loss: 0.0063, Val Loss: 0.0064
Epoch 447/1000, Train Loss: 0.0062, Val Loss: 0.0065
Epoch 448/1000, Train Loss: 0.0062, Val Loss: 0.0061
Epoch 449/1000, Train Loss: 0.0061, Val Loss: 0.0059
Epoch 450/1000, Train Loss: 0.0060, Val Loss: 0.0060
模型已保存至 ./checkpoints/checkpoint_epoch_450.pth
Epoch 451/1000, Train Loss: 0.0060, Val Loss: 0.0064
Epoch 452/1000, Train Loss: 0.0060, Val Loss: 0.0060
Epoch 453/1000, Train Loss: 0.0059, Val Loss: 0.0059
Epoch 454/1000, Train Loss: 0.0059, Val Loss: 0.0059
Epoch 455/1000, Train Loss: 0.0059, Val Loss: 0.0060
模型已保存至 ./checkpoints/checkpoint_epoch_455.pth
Epoch 456/1000, Train Loss: 0.0058, Val Loss: 0.0060
Epoch 457/1000, Train Loss: 0.0058, Val Loss: 0.0058
Epoch 458/1000, Train Loss: 0.0058, Val Loss: 0.0059
Epoch 459/1000, Train Loss: 0.0057, Val Loss: 0.0058
Epoch 460/1000, Train Loss: 0.0057, Val Loss: 0.0059
模型已保存至 ./checkpoints/checkpoint_epoch_460.pth
Epoch 461/1000, Train Loss: 0.0057, Val Loss: 0.0058
Epoch 462/1000, Train Loss: 0.0056, Val Loss: 0.0058
Epoch 463/1000, Train Loss: 0.0056, Val Loss: 0.0058
Epoch 464/1000, Train Loss: 0.0056, Val Loss: 0.0058
Epoch 465/1000, Train Loss: 0.0055, Val Loss: 0.0058
模型已保存至 ./checkpoints/checkpoint_epoch_465.pth
Epoch 466/1000, Train Loss: 0.0055, Val Loss: 0.0057
Epoch 467/1000, Train Loss: 0.0055, Val Loss: 0.0057
Epoch 468/1000, Train Loss: 0.0055, Val Loss: 0.0056
Epoch 469/1000, Train Loss: 0.0054, Val Loss: 0.0056
Epoch 470/1000, Train Loss: 0.0054, Val Loss: 0.0056
模型已保存至 ./checkpoints/checkpoint_epoch_470.pth
Epoch 471/1000, Train Loss: 0.0054, Val Loss: 0.0057
Epoch 472/1000, Train Loss: 0.0054, Val Loss: 0.0056
Epoch 473/1000, Train Loss: 0.0055, Val Loss: 0.0060
Epoch 474/1000, Train Loss: 0.0058, Val Loss: 0.0063
Epoch 475/1000, Train Loss: 0.0066, Val Loss: 0.0076
模型已保存至 ./checkpoints/checkpoint_epoch_475.pth
Epoch 476/1000, Train Loss: 0.0080, Val Loss: 0.0086
Epoch 477/1000, Train Loss: 0.0075, Val Loss: 0.0053
Epoch 478/1000, Train Loss: 0.0056, Val Loss: 0.0061
Epoch 479/1000, Train Loss: 0.0064, Val Loss: 0.0078
Epoch 480/1000, Train Loss: 0.0064, Val Loss: 0.0070
模型已保存至 ./checkpoints/checkpoint_epoch_480.pth
Epoch 481/1000, Train Loss: 0.0057, Val Loss: 0.0060
Epoch 482/1000, Train Loss: 0.0061, Val Loss: 0.0057
Epoch 483/1000, Train Loss: 0.0057, Val Loss: 0.0062
Epoch 484/1000, Train Loss: 0.0059, Val Loss: 0.0062
Epoch 485/1000, Train Loss: 0.0055, Val Loss: 0.0067
模型已保存至 ./checkpoints/checkpoint_epoch_485.pth
Epoch 486/1000, Train Loss: 0.0058, Val Loss: 0.0058
Epoch 487/1000, Train Loss: 0.0053, Val Loss: 0.0062
Epoch 488/1000, Train Loss: 0.0056, Val Loss: 0.0058
Epoch 489/1000, Train Loss: 0.0053, Val Loss: 0.0059
Epoch 490/1000, Train Loss: 0.0054, Val Loss: 0.0060
模型已保存至 ./checkpoints/checkpoint_epoch_490.pth
Epoch 491/1000, Train Loss: 0.0053, Val Loss: 0.0057
Epoch 492/1000, Train Loss: 0.0053, Val Loss: 0.0057
Epoch 493/1000, Train Loss: 0.0052, Val Loss: 0.0055
Epoch 494/1000, Train Loss: 0.0052, Val Loss: 0.0054
Epoch 495/1000, Train Loss: 0.0051, Val Loss: 0.0055
模型已保存至 ./checkpoints/checkpoint_epoch_495.pth
Epoch 496/1000, Train Loss: 0.0051, Val Loss: 0.0053
Epoch 497/1000, Train Loss: 0.0051, Val Loss: 0.0053
Epoch 498/1000, Train Loss: 0.0050, Val Loss: 0.0052
Epoch 499/1000, Train Loss: 0.0050, Val Loss: 0.0051
Epoch 500/1000, Train Loss: 0.0050, Val Loss: 0.0052
模型已保存至 ./checkpoints/checkpoint_epoch_500.pth
Epoch 501/1000, Train Loss: 0.0050, Val Loss: 0.0052
Epoch 502/1000, Train Loss: 0.0049, Val Loss: 0.0051
Epoch 503/1000, Train Loss: 0.0049, Val Loss: 0.0051
Epoch 504/1000, Train Loss: 0.0049, Val Loss: 0.0050
Epoch 505/1000, Train Loss: 0.0049, Val Loss: 0.0050
模型已保存至 ./checkpoints/checkpoint_epoch_505.pth
Epoch 506/1000, Train Loss: 0.0048, Val Loss: 0.0051
Epoch 507/1000, Train Loss: 0.0048, Val Loss: 0.0050
Epoch 508/1000, Train Loss: 0.0048, Val Loss: 0.0049
Epoch 509/1000, Train Loss: 0.0048, Val Loss: 0.0049
Epoch 510/1000, Train Loss: 0.0047, Val Loss: 0.0049
模型已保存至 ./checkpoints/checkpoint_epoch_510.pth
Epoch 511/1000, Train Loss: 0.0047, Val Loss: 0.0049
Epoch 512/1000, Train Loss: 0.0047, Val Loss: 0.0049
Epoch 513/1000, Train Loss: 0.0047, Val Loss: 0.0048
Epoch 514/1000, Train Loss: 0.0046, Val Loss: 0.0048
Epoch 515/1000, Train Loss: 0.0046, Val Loss: 0.0048
模型已保存至 ./checkpoints/checkpoint_epoch_515.pth
Epoch 516/1000, Train Loss: 0.0046, Val Loss: 0.0048
Epoch 517/1000, Train Loss: 0.0046, Val Loss: 0.0048
Epoch 518/1000, Train Loss: 0.0046, Val Loss: 0.0048
Epoch 519/1000, Train Loss: 0.0045, Val Loss: 0.0047
Epoch 520/1000, Train Loss: 0.0045, Val Loss: 0.0047
模型已保存至 ./checkpoints/checkpoint_epoch_520.pth
Epoch 521/1000, Train Loss: 0.0045, Val Loss: 0.0047
Epoch 522/1000, Train Loss: 0.0045, Val Loss: 0.0046
Epoch 523/1000, Train Loss: 0.0045, Val Loss: 0.0046
Epoch 524/1000, Train Loss: 0.0045, Val Loss: 0.0046
Epoch 525/1000, Train Loss: 0.0044, Val Loss: 0.0046
模型已保存至 ./checkpoints/checkpoint_epoch_525.pth
Epoch 526/1000, Train Loss: 0.0044, Val Loss: 0.0046
Epoch 527/1000, Train Loss: 0.0044, Val Loss: 0.0046
Epoch 528/1000, Train Loss: 0.0044, Val Loss: 0.0045
Epoch 529/1000, Train Loss: 0.0044, Val Loss: 0.0045
Epoch 530/1000, Train Loss: 0.0044, Val Loss: 0.0045
模型已保存至 ./checkpoints/checkpoint_epoch_530.pth
Epoch 531/1000, Train Loss: 0.0044, Val Loss: 0.0046
Epoch 532/1000, Train Loss: 0.0045, Val Loss: 0.0045
Epoch 533/1000, Train Loss: 0.0046, Val Loss: 0.0046
Epoch 534/1000, Train Loss: 0.0047, Val Loss: 0.0044
Epoch 535/1000, Train Loss: 0.0045, Val Loss: 0.0044
模型已保存至 ./checkpoints/checkpoint_epoch_535.pth
Epoch 536/1000, Train Loss: 0.0044, Val Loss: 0.0044
Epoch 537/1000, Train Loss: 0.0045, Val Loss: 0.0043
Epoch 538/1000, Train Loss: 0.0045, Val Loss: 0.0041
Epoch 539/1000, Train Loss: 0.0043, Val Loss: 0.0042
Epoch 540/1000, Train Loss: 0.0043, Val Loss: 0.0041
模型已保存至 ./checkpoints/checkpoint_epoch_540.pth
Epoch 541/1000, Train Loss: 0.0043, Val Loss: 0.0042
Epoch 542/1000, Train Loss: 0.0043, Val Loss: 0.0043
Epoch 543/1000, Train Loss: 0.0043, Val Loss: 0.0042
Epoch 544/1000, Train Loss: 0.0042, Val Loss: 0.0041
Epoch 545/1000, Train Loss: 0.0042, Val Loss: 0.0043
模型已保存至 ./checkpoints/checkpoint_epoch_545.pth
Epoch 546/1000, Train Loss: 0.0042, Val Loss: 0.0042
Epoch 547/1000, Train Loss: 0.0041, Val Loss: 0.0041
Epoch 548/1000, Train Loss: 0.0042, Val Loss: 0.0041
Epoch 549/1000, Train Loss: 0.0041, Val Loss: 0.0041
Epoch 550/1000, Train Loss: 0.0041, Val Loss: 0.0040
模型已保存至 ./checkpoints/checkpoint_epoch_550.pth
Epoch 551/1000, Train Loss: 0.0041, Val Loss: 0.0041
Epoch 552/1000, Train Loss: 0.0041, Val Loss: 0.0041
Epoch 553/1000, Train Loss: 0.0041, Val Loss: 0.0040
Epoch 554/1000, Train Loss: 0.0040, Val Loss: 0.0040
Epoch 555/1000, Train Loss: 0.0040, Val Loss: 0.0040
模型已保存至 ./checkpoints/checkpoint_epoch_555.pth
Epoch 556/1000, Train Loss: 0.0040, Val Loss: 0.0040
Epoch 557/1000, Train Loss: 0.0040, Val Loss: 0.0040
Epoch 558/1000, Train Loss: 0.0040, Val Loss: 0.0040
Epoch 559/1000, Train Loss: 0.0040, Val Loss: 0.0040
Epoch 560/1000, Train Loss: 0.0039, Val Loss: 0.0040
模型已保存至 ./checkpoints/checkpoint_epoch_560.pth
Epoch 561/1000, Train Loss: 0.0039, Val Loss: 0.0040
Epoch 562/1000, Train Loss: 0.0039, Val Loss: 0.0039
Epoch 563/1000, Train Loss: 0.0039, Val Loss: 0.0040
Epoch 564/1000, Train Loss: 0.0039, Val Loss: 0.0040
Epoch 565/1000, Train Loss: 0.0039, Val Loss: 0.0039
模型已保存至 ./checkpoints/checkpoint_epoch_565.pth
Epoch 566/1000, Train Loss: 0.0039, Val Loss: 0.0040
Epoch 567/1000, Train Loss: 0.0039, Val Loss: 0.0039
Epoch 568/1000, Train Loss: 0.0038, Val Loss: 0.0040
Epoch 569/1000, Train Loss: 0.0039, Val Loss: 0.0041
Epoch 570/1000, Train Loss: 0.0039, Val Loss: 0.0042
模型已保存至 ./checkpoints/checkpoint_epoch_570.pth
Epoch 571/1000, Train Loss: 0.0041, Val Loss: 0.0044
Epoch 572/1000, Train Loss: 0.0043, Val Loss: 0.0045
Epoch 573/1000, Train Loss: 0.0044, Val Loss: 0.0041
Epoch 574/1000, Train Loss: 0.0042, Val Loss: 0.0038
Epoch 575/1000, Train Loss: 0.0039, Val Loss: 0.0038
模型已保存至 ./checkpoints/checkpoint_epoch_575.pth
Epoch 576/1000, Train Loss: 0.0039, Val Loss: 0.0040
Epoch 577/1000, Train Loss: 0.0040, Val Loss: 0.0040
Epoch 578/1000, Train Loss: 0.0039, Val Loss: 0.0038
Epoch 579/1000, Train Loss: 0.0038, Val Loss: 0.0040
Epoch 580/1000, Train Loss: 0.0039, Val Loss: 0.0040
模型已保存至 ./checkpoints/checkpoint_epoch_580.pth
Epoch 581/1000, Train Loss: 0.0038, Val Loss: 0.0038
Epoch 582/1000, Train Loss: 0.0037, Val Loss: 0.0039
Epoch 583/1000, Train Loss: 0.0038, Val Loss: 0.0038
Epoch 584/1000, Train Loss: 0.0037, Val Loss: 0.0038
Epoch 585/1000, Train Loss: 0.0037, Val Loss: 0.0038
模型已保存至 ./checkpoints/checkpoint_epoch_585.pth
Epoch 586/1000, Train Loss: 0.0037, Val Loss: 0.0038
Epoch 587/1000, Train Loss: 0.0037, Val Loss: 0.0037
Epoch 588/1000, Train Loss: 0.0036, Val Loss: 0.0037
Epoch 589/1000, Train Loss: 0.0036, Val Loss: 0.0037
Epoch 590/1000, Train Loss: 0.0036, Val Loss: 0.0037
模型已保存至 ./checkpoints/checkpoint_epoch_590.pth
Epoch 591/1000, Train Loss: 0.0036, Val Loss: 0.0037
Epoch 592/1000, Train Loss: 0.0036, Val Loss: 0.0037
Epoch 593/1000, Train Loss: 0.0036, Val Loss: 0.0037
Epoch 594/1000, Train Loss: 0.0035, Val Loss: 0.0037
Epoch 595/1000, Train Loss: 0.0036, Val Loss: 0.0037
模型已保存至 ./checkpoints/checkpoint_epoch_595.pth
Epoch 596/1000, Train Loss: 0.0035, Val Loss: 0.0036
Epoch 597/1000, Train Loss: 0.0035, Val Loss: 0.0036
Epoch 598/1000, Train Loss: 0.0035, Val Loss: 0.0036
Epoch 599/1000, Train Loss: 0.0035, Val Loss: 0.0036
Epoch 600/1000, Train Loss: 0.0035, Val Loss: 0.0036
模型已保存至 ./checkpoints/checkpoint_epoch_600.pth
Epoch 601/1000, Train Loss: 0.0035, Val Loss: 0.0038
Epoch 602/1000, Train Loss: 0.0036, Val Loss: 0.0039
Epoch 603/1000, Train Loss: 0.0038, Val Loss: 0.0043
Epoch 604/1000, Train Loss: 0.0042, Val Loss: 0.0043
Epoch 605/1000, Train Loss: 0.0044, Val Loss: 0.0037
模型已保存至 ./checkpoints/checkpoint_epoch_605.pth
Epoch 606/1000, Train Loss: 0.0039, Val Loss: 0.0035
Epoch 607/1000, Train Loss: 0.0035, Val Loss: 0.0035
Epoch 608/1000, Train Loss: 0.0037, Val Loss: 0.0037
Epoch 609/1000, Train Loss: 0.0037, Val Loss: 0.0035
Epoch 610/1000, Train Loss: 0.0036, Val Loss: 0.0035
模型已保存至 ./checkpoints/checkpoint_epoch_610.pth
Epoch 611/1000, Train Loss: 0.0035, Val Loss: 0.0037
Epoch 612/1000, Train Loss: 0.0036, Val Loss: 0.0035
Epoch 613/1000, Train Loss: 0.0035, Val Loss: 0.0035
Epoch 614/1000, Train Loss: 0.0035, Val Loss: 0.0035
Epoch 615/1000, Train Loss: 0.0034, Val Loss: 0.0035
模型已保存至 ./checkpoints/checkpoint_epoch_615.pth
Epoch 616/1000, Train Loss: 0.0034, Val Loss: 0.0035
Epoch 617/1000, Train Loss: 0.0034, Val Loss: 0.0034
Epoch 618/1000, Train Loss: 0.0034, Val Loss: 0.0034
Epoch 619/1000, Train Loss: 0.0034, Val Loss: 0.0033
Epoch 620/1000, Train Loss: 0.0034, Val Loss: 0.0033
模型已保存至 ./checkpoints/checkpoint_epoch_620.pth
Epoch 621/1000, Train Loss: 0.0033, Val Loss: 0.0033
Epoch 622/1000, Train Loss: 0.0033, Val Loss: 0.0033
Epoch 623/1000, Train Loss: 0.0033, Val Loss: 0.0034
Epoch 624/1000, Train Loss: 0.0033, Val Loss: 0.0033
Epoch 625/1000, Train Loss: 0.0033, Val Loss: 0.0033
模型已保存至 ./checkpoints/checkpoint_epoch_625.pth
Epoch 626/1000, Train Loss: 0.0033, Val Loss: 0.0033
Epoch 627/1000, Train Loss: 0.0032, Val Loss: 0.0033
Epoch 628/1000, Train Loss: 0.0032, Val Loss: 0.0033
Epoch 629/1000, Train Loss: 0.0032, Val Loss: 0.0032
Epoch 630/1000, Train Loss: 0.0032, Val Loss: 0.0033
模型已保存至 ./checkpoints/checkpoint_epoch_630.pth
Epoch 631/1000, Train Loss: 0.0032, Val Loss: 0.0033
Epoch 632/1000, Train Loss: 0.0032, Val Loss: 0.0032
Epoch 633/1000, Train Loss: 0.0032, Val Loss: 0.0033
Epoch 634/1000, Train Loss: 0.0032, Val Loss: 0.0032
Epoch 635/1000, Train Loss: 0.0031, Val Loss: 0.0032
模型已保存至 ./checkpoints/checkpoint_epoch_635.pth
Epoch 636/1000, Train Loss: 0.0031, Val Loss: 0.0032
Epoch 637/1000, Train Loss: 0.0031, Val Loss: 0.0032
Epoch 638/1000, Train Loss: 0.0031, Val Loss: 0.0032
Epoch 639/1000, Train Loss: 0.0031, Val Loss: 0.0031
Epoch 640/1000, Train Loss: 0.0031, Val Loss: 0.0032
模型已保存至 ./checkpoints/checkpoint_epoch_640.pth
Epoch 641/1000, Train Loss: 0.0031, Val Loss: 0.0031
Epoch 642/1000, Train Loss: 0.0031, Val Loss: 0.0032
Epoch 643/1000, Train Loss: 0.0031, Val Loss: 0.0031
Epoch 644/1000, Train Loss: 0.0031, Val Loss: 0.0032
Epoch 645/1000, Train Loss: 0.0031, Val Loss: 0.0031
模型已保存至 ./checkpoints/checkpoint_epoch_645.pth
Epoch 646/1000, Train Loss: 0.0032, Val Loss: 0.0033
Epoch 647/1000, Train Loss: 0.0033, Val Loss: 0.0032
Epoch 648/1000, Train Loss: 0.0033, Val Loss: 0.0030
Epoch 649/1000, Train Loss: 0.0031, Val Loss: 0.0029
Epoch 650/1000, Train Loss: 0.0031, Val Loss: 0.0028
模型已保存至 ./checkpoints/checkpoint_epoch_650.pth
Epoch 651/1000, Train Loss: 0.0032, Val Loss: 0.0030
Epoch 652/1000, Train Loss: 0.0032, Val Loss: 0.0028
Epoch 653/1000, Train Loss: 0.0030, Val Loss: 0.0029
Epoch 654/1000, Train Loss: 0.0031, Val Loss: 0.0030
Epoch 655/1000, Train Loss: 0.0031, Val Loss: 0.0029
模型已保存至 ./checkpoints/checkpoint_epoch_655.pth
Epoch 656/1000, Train Loss: 0.0030, Val Loss: 0.0029
Epoch 657/1000, Train Loss: 0.0031, Val Loss: 0.0030
Epoch 658/1000, Train Loss: 0.0030, Val Loss: 0.0028
Epoch 659/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 660/1000, Train Loss: 0.0030, Val Loss: 0.0028
模型已保存至 ./checkpoints/checkpoint_epoch_660.pth
Epoch 661/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 662/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 663/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 664/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 665/1000, Train Loss: 0.0029, Val Loss: 0.0028
模型已保存至 ./checkpoints/checkpoint_epoch_665.pth
Epoch 666/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 667/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 668/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 669/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 670/1000, Train Loss: 0.0028, Val Loss: 0.0029
模型已保存至 ./checkpoints/checkpoint_epoch_670.pth
Epoch 671/1000, Train Loss: 0.0028, Val Loss: 0.0028
Epoch 672/1000, Train Loss: 0.0029, Val Loss: 0.0029
Epoch 673/1000, Train Loss: 0.0029, Val Loss: 0.0030
Epoch 674/1000, Train Loss: 0.0030, Val Loss: 0.0032
Epoch 675/1000, Train Loss: 0.0031, Val Loss: 0.0032
模型已保存至 ./checkpoints/checkpoint_epoch_675.pth
Epoch 676/1000, Train Loss: 0.0033, Val Loss: 0.0033
Epoch 677/1000, Train Loss: 0.0033, Val Loss: 0.0030
Epoch 678/1000, Train Loss: 0.0031, Val Loss: 0.0030
Epoch 679/1000, Train Loss: 0.0029, Val Loss: 0.0029
Epoch 680/1000, Train Loss: 0.0029, Val Loss: 0.0029
模型已保存至 ./checkpoints/checkpoint_epoch_680.pth
Epoch 681/1000, Train Loss: 0.0030, Val Loss: 0.0029
Epoch 682/1000, Train Loss: 0.0028, Val Loss: 0.0028
Epoch 683/1000, Train Loss: 0.0028, Val Loss: 0.0029
Epoch 684/1000, Train Loss: 0.0029, Val Loss: 0.0028
Epoch 685/1000, Train Loss: 0.0028, Val Loss: 0.0028
模型已保存至 ./checkpoints/checkpoint_epoch_685.pth
Epoch 686/1000, Train Loss: 0.0027, Val Loss: 0.0028
Epoch 687/1000, Train Loss: 0.0028, Val Loss: 0.0028
Epoch 688/1000, Train Loss: 0.0028, Val Loss: 0.0029
Epoch 689/1000, Train Loss: 0.0027, Val Loss: 0.0028
Epoch 690/1000, Train Loss: 0.0028, Val Loss: 0.0028
模型已保存至 ./checkpoints/checkpoint_epoch_690.pth
Epoch 691/1000, Train Loss: 0.0027, Val Loss: 0.0028
Epoch 692/1000, Train Loss: 0.0027, Val Loss: 0.0027
Epoch 693/1000, Train Loss: 0.0027, Val Loss: 0.0027
Epoch 694/1000, Train Loss: 0.0027, Val Loss: 0.0027
Epoch 695/1000, Train Loss: 0.0027, Val Loss: 0.0027
模型已保存至 ./checkpoints/checkpoint_epoch_695.pth
Epoch 696/1000, Train Loss: 0.0027, Val Loss: 0.0027
Epoch 697/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 698/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 699/1000, Train Loss: 0.0027, Val Loss: 0.0027
Epoch 700/1000, Train Loss: 0.0026, Val Loss: 0.0027
模型已保存至 ./checkpoints/checkpoint_epoch_700.pth
Epoch 701/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 702/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 703/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 704/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 705/1000, Train Loss: 0.0026, Val Loss: 0.0027
模型已保存至 ./checkpoints/checkpoint_epoch_705.pth
Epoch 706/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 707/1000, Train Loss: 0.0026, Val Loss: 0.0027
Epoch 708/1000, Train Loss: 0.0026, Val Loss: 0.0026
Epoch 709/1000, Train Loss: 0.0027, Val Loss: 0.0028
Epoch 710/1000, Train Loss: 0.0027, Val Loss: 0.0028
模型已保存至 ./checkpoints/checkpoint_epoch_710.pth
Epoch 711/1000, Train Loss: 0.0027, Val Loss: 0.0030
Epoch 712/1000, Train Loss: 0.0028, Val Loss: 0.0029
Epoch 713/1000, Train Loss: 0.0028, Val Loss: 0.0027
Epoch 714/1000, Train Loss: 0.0027, Val Loss: 0.0026
Epoch 715/1000, Train Loss: 0.0026, Val Loss: 0.0026
模型已保存至 ./checkpoints/checkpoint_epoch_715.pth
Epoch 716/1000, Train Loss: 0.0026, Val Loss: 0.0026
Epoch 717/1000, Train Loss: 0.0026, Val Loss: 0.0026
Epoch 718/1000, Train Loss: 0.0026, Val Loss: 0.0026
Epoch 719/1000, Train Loss: 0.0025, Val Loss: 0.0026
Epoch 720/1000, Train Loss: 0.0026, Val Loss: 0.0025
模型已保存至 ./checkpoints/checkpoint_epoch_720.pth
Epoch 721/1000, Train Loss: 0.0026, Val Loss: 0.0025
Epoch 722/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 723/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 724/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 725/1000, Train Loss: 0.0025, Val Loss: 0.0025
模型已保存至 ./checkpoints/checkpoint_epoch_725.pth
Epoch 726/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 727/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 728/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 729/1000, Train Loss: 0.0025, Val Loss: 0.0024
Epoch 730/1000, Train Loss: 0.0024, Val Loss: 0.0025
模型已保存至 ./checkpoints/checkpoint_epoch_730.pth
Epoch 731/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 732/1000, Train Loss: 0.0024, Val Loss: 0.0024
Epoch 733/1000, Train Loss: 0.0024, Val Loss: 0.0024
Epoch 734/1000, Train Loss: 0.0024, Val Loss: 0.0024
Epoch 735/1000, Train Loss: 0.0024, Val Loss: 0.0025
模型已保存至 ./checkpoints/checkpoint_epoch_735.pth
Epoch 736/1000, Train Loss: 0.0024, Val Loss: 0.0024
Epoch 737/1000, Train Loss: 0.0024, Val Loss: 0.0025
Epoch 738/1000, Train Loss: 0.0024, Val Loss: 0.0025
Epoch 739/1000, Train Loss: 0.0024, Val Loss: 0.0026
Epoch 740/1000, Train Loss: 0.0025, Val Loss: 0.0025
模型已保存至 ./checkpoints/checkpoint_epoch_740.pth
Epoch 741/1000, Train Loss: 0.0025, Val Loss: 0.0026
Epoch 742/1000, Train Loss: 0.0025, Val Loss: 0.0025
Epoch 743/1000, Train Loss: 0.0025, Val Loss: 0.0026
Epoch 744/1000, Train Loss: 0.0025, Val Loss: 0.0024
Epoch 745/1000, Train Loss: 0.0024, Val Loss: 0.0024
模型已保存至 ./checkpoints/checkpoint_epoch_745.pth
Epoch 746/1000, Train Loss: 0.0024, Val Loss: 0.0025
Epoch 747/1000, Train Loss: 0.0025, Val Loss: 0.0024
Epoch 748/1000, Train Loss: 0.0025, Val Loss: 0.0024
Epoch 749/1000, Train Loss: 0.0024, Val Loss: 0.0023
Epoch 750/1000, Train Loss: 0.0024, Val Loss: 0.0023
模型已保存至 ./checkpoints/checkpoint_epoch_750.pth
Epoch 751/1000, Train Loss: 0.0024, Val Loss: 0.0024
Epoch 752/1000, Train Loss: 0.0024, Val Loss: 0.0022
Epoch 753/1000, Train Loss: 0.0023, Val Loss: 0.0023
Epoch 754/1000, Train Loss: 0.0023, Val Loss: 0.0023
Epoch 755/1000, Train Loss: 0.0023, Val Loss: 0.0023
模型已保存至 ./checkpoints/checkpoint_epoch_755.pth
Epoch 756/1000, Train Loss: 0.0023, Val Loss: 0.0023
Epoch 757/1000, Train Loss: 0.0023, Val Loss: 0.0022
Epoch 758/1000, Train Loss: 0.0023, Val Loss: 0.0023
Epoch 759/1000, Train Loss: 0.0023, Val Loss: 0.0022
Epoch 760/1000, Train Loss: 0.0022, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_760.pth
Epoch 761/1000, Train Loss: 0.0022, Val Loss: 0.0023
Epoch 762/1000, Train Loss: 0.0023, Val Loss: 0.0022
Epoch 763/1000, Train Loss: 0.0023, Val Loss: 0.0022
Epoch 764/1000, Train Loss: 0.0022, Val Loss: 0.0022
Epoch 765/1000, Train Loss: 0.0022, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_765.pth
Epoch 766/1000, Train Loss: 0.0022, Val Loss: 0.0023
Epoch 767/1000, Train Loss: 0.0022, Val Loss: 0.0022
Epoch 768/1000, Train Loss: 0.0022, Val Loss: 0.0022
Epoch 769/1000, Train Loss: 0.0022, Val Loss: 0.0023
Epoch 770/1000, Train Loss: 0.0022, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_770.pth
Epoch 771/1000, Train Loss: 0.0022, Val Loss: 0.0022
Epoch 772/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 773/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 774/1000, Train Loss: 0.0022, Val Loss: 0.0023
Epoch 775/1000, Train Loss: 0.0022, Val Loss: 0.0024
模型已保存至 ./checkpoints/checkpoint_epoch_775.pth
Epoch 776/1000, Train Loss: 0.0024, Val Loss: 0.0025
Epoch 777/1000, Train Loss: 0.0026, Val Loss: 0.0026
Epoch 778/1000, Train Loss: 0.0025, Val Loss: 0.0022
Epoch 779/1000, Train Loss: 0.0026, Val Loss: 0.0028
Epoch 780/1000, Train Loss: 0.0026, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_780.pth
Epoch 781/1000, Train Loss: 0.0023, Val Loss: 0.0021
Epoch 782/1000, Train Loss: 0.0023, Val Loss: 0.0025
Epoch 783/1000, Train Loss: 0.0024, Val Loss: 0.0021
Epoch 784/1000, Train Loss: 0.0023, Val Loss: 0.0023
Epoch 785/1000, Train Loss: 0.0023, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_785.pth
Epoch 786/1000, Train Loss: 0.0023, Val Loss: 0.0022
Epoch 787/1000, Train Loss: 0.0022, Val Loss: 0.0022
Epoch 788/1000, Train Loss: 0.0022, Val Loss: 0.0021
Epoch 789/1000, Train Loss: 0.0022, Val Loss: 0.0022
Epoch 790/1000, Train Loss: 0.0022, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_790.pth
Epoch 791/1000, Train Loss: 0.0022, Val Loss: 0.0021
Epoch 792/1000, Train Loss: 0.0022, Val Loss: 0.0021
Epoch 793/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 794/1000, Train Loss: 0.0021, Val Loss: 0.0021
Epoch 795/1000, Train Loss: 0.0021, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_795.pth
Epoch 796/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 797/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 798/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 799/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 800/1000, Train Loss: 0.0021, Val Loss: 0.0023
模型已保存至 ./checkpoints/checkpoint_epoch_800.pth
Epoch 801/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 802/1000, Train Loss: 0.0022, Val Loss: 0.0023
Epoch 803/1000, Train Loss: 0.0022, Val Loss: 0.0023
Epoch 804/1000, Train Loss: 0.0023, Val Loss: 0.0024
Epoch 805/1000, Train Loss: 0.0023, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_805.pth
Epoch 806/1000, Train Loss: 0.0022, Val Loss: 0.0021
Epoch 807/1000, Train Loss: 0.0021, Val Loss: 0.0021
Epoch 808/1000, Train Loss: 0.0020, Val Loss: 0.0021
Epoch 809/1000, Train Loss: 0.0021, Val Loss: 0.0022
Epoch 810/1000, Train Loss: 0.0021, Val Loss: 0.0022
模型已保存至 ./checkpoints/checkpoint_epoch_810.pth
Epoch 811/1000, Train Loss: 0.0020, Val Loss: 0.0022
Epoch 812/1000, Train Loss: 0.0020, Val Loss: 0.0022
Epoch 813/1000, Train Loss: 0.0020, Val Loss: 0.0022
Epoch 814/1000, Train Loss: 0.0020, Val Loss: 0.0022
Epoch 815/1000, Train Loss: 0.0020, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_815.pth
Epoch 816/1000, Train Loss: 0.0020, Val Loss: 0.0022
Epoch 817/1000, Train Loss: 0.0020, Val Loss: 0.0022
Epoch 818/1000, Train Loss: 0.0020, Val Loss: 0.0021
Epoch 819/1000, Train Loss: 0.0020, Val Loss: 0.0021
Epoch 820/1000, Train Loss: 0.0019, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_820.pth
Epoch 821/1000, Train Loss: 0.0019, Val Loss: 0.0021
Epoch 822/1000, Train Loss: 0.0019, Val Loss: 0.0021
Epoch 823/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 824/1000, Train Loss: 0.0019, Val Loss: 0.0021
Epoch 825/1000, Train Loss: 0.0019, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_825.pth
Epoch 826/1000, Train Loss: 0.0019, Val Loss: 0.0021
Epoch 827/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 828/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 829/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 830/1000, Train Loss: 0.0019, Val Loss: 0.0020
模型已保存至 ./checkpoints/checkpoint_epoch_830.pth
Epoch 831/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 832/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 833/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 834/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 835/1000, Train Loss: 0.0019, Val Loss: 0.0020
模型已保存至 ./checkpoints/checkpoint_epoch_835.pth
Epoch 836/1000, Train Loss: 0.0019, Val Loss: 0.0020
Epoch 837/1000, Train Loss: 0.0020, Val Loss: 0.0020
Epoch 838/1000, Train Loss: 0.0020, Val Loss: 0.0019
Epoch 839/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 840/1000, Train Loss: 0.0019, Val Loss: 0.0019
模型已保存至 ./checkpoints/checkpoint_epoch_840.pth
Epoch 841/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 842/1000, Train Loss: 0.0019, Val Loss: 0.0018
Epoch 843/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 844/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 845/1000, Train Loss: 0.0020, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_845.pth
Epoch 846/1000, Train Loss: 0.0021, Val Loss: 0.0020
Epoch 847/1000, Train Loss: 0.0021, Val Loss: 0.0021
Epoch 848/1000, Train Loss: 0.0023, Val Loss: 0.0022
Epoch 849/1000, Train Loss: 0.0025, Val Loss: 0.0022
Epoch 850/1000, Train Loss: 0.0025, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_850.pth
Epoch 851/1000, Train Loss: 0.0022, Val Loss: 0.0026
Epoch 852/1000, Train Loss: 0.0023, Val Loss: 0.0021
Epoch 853/1000, Train Loss: 0.0022, Val Loss: 0.0020
Epoch 854/1000, Train Loss: 0.0020, Val Loss: 0.0024
Epoch 855/1000, Train Loss: 0.0022, Val Loss: 0.0018
模型已保存至 ./checkpoints/checkpoint_epoch_855.pth
Epoch 856/1000, Train Loss: 0.0020, Val Loss: 0.0019
Epoch 857/1000, Train Loss: 0.0020, Val Loss: 0.0019
Epoch 858/1000, Train Loss: 0.0020, Val Loss: 0.0019
Epoch 859/1000, Train Loss: 0.0020, Val Loss: 0.0019
Epoch 860/1000, Train Loss: 0.0020, Val Loss: 0.0019
模型已保存至 ./checkpoints/checkpoint_epoch_860.pth
Epoch 861/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 862/1000, Train Loss: 0.0019, Val Loss: 0.0023
Epoch 863/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 864/1000, Train Loss: 0.0018, Val Loss: 0.0021
Epoch 865/1000, Train Loss: 0.0019, Val Loss: 0.0019
模型已保存至 ./checkpoints/checkpoint_epoch_865.pth
Epoch 866/1000, Train Loss: 0.0018, Val Loss: 0.0020
Epoch 867/1000, Train Loss: 0.0018, Val Loss: 0.0018
Epoch 868/1000, Train Loss: 0.0018, Val Loss: 0.0019
Epoch 869/1000, Train Loss: 0.0018, Val Loss: 0.0018
Epoch 870/1000, Train Loss: 0.0018, Val Loss: 0.0019
模型已保存至 ./checkpoints/checkpoint_epoch_870.pth
Epoch 871/1000, Train Loss: 0.0018, Val Loss: 0.0019
Epoch 872/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 873/1000, Train Loss: 0.0017, Val Loss: 0.0018
Epoch 874/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 875/1000, Train Loss: 0.0017, Val Loss: 0.0019
模型已保存至 ./checkpoints/checkpoint_epoch_875.pth
Epoch 876/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 877/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 878/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 879/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 880/1000, Train Loss: 0.0017, Val Loss: 0.0018
模型已保存至 ./checkpoints/checkpoint_epoch_880.pth
Epoch 881/1000, Train Loss: 0.0017, Val Loss: 0.0018
Epoch 882/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 883/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 884/1000, Train Loss: 0.0017, Val Loss: 0.0018
Epoch 885/1000, Train Loss: 0.0017, Val Loss: 0.0018
模型已保存至 ./checkpoints/checkpoint_epoch_885.pth
Epoch 886/1000, Train Loss: 0.0017, Val Loss: 0.0018
Epoch 887/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 888/1000, Train Loss: 0.0017, Val Loss: 0.0019
Epoch 889/1000, Train Loss: 0.0017, Val Loss: 0.0020
Epoch 890/1000, Train Loss: 0.0018, Val Loss: 0.0021
模型已保存至 ./checkpoints/checkpoint_epoch_890.pth
Epoch 891/1000, Train Loss: 0.0020, Val Loss: 0.0023
Epoch 892/1000, Train Loss: 0.0022, Val Loss: 0.0024
Epoch 893/1000, Train Loss: 0.0023, Val Loss: 0.0020
Epoch 894/1000, Train Loss: 0.0021, Val Loss: 0.0018
Epoch 895/1000, Train Loss: 0.0019, Val Loss: 0.0020
模型已保存至 ./checkpoints/checkpoint_epoch_895.pth
Epoch 896/1000, Train Loss: 0.0020, Val Loss: 0.0017
Epoch 897/1000, Train Loss: 0.0019, Val Loss: 0.0018
Epoch 898/1000, Train Loss: 0.0019, Val Loss: 0.0018
Epoch 899/1000, Train Loss: 0.0019, Val Loss: 0.0018
Epoch 900/1000, Train Loss: 0.0019, Val Loss: 0.0018
模型已保存至 ./checkpoints/checkpoint_epoch_900.pth
Epoch 901/1000, Train Loss: 0.0019, Val Loss: 0.0018
Epoch 902/1000, Train Loss: 0.0018, Val Loss: 0.0016
Epoch 903/1000, Train Loss: 0.0018, Val Loss: 0.0017
Epoch 904/1000, Train Loss: 0.0018, Val Loss: 0.0016
Epoch 905/1000, Train Loss: 0.0017, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_905.pth
Epoch 906/1000, Train Loss: 0.0017, Val Loss: 0.0017
Epoch 907/1000, Train Loss: 0.0017, Val Loss: 0.0016
Epoch 908/1000, Train Loss: 0.0017, Val Loss: 0.0016
Epoch 909/1000, Train Loss: 0.0017, Val Loss: 0.0017
Epoch 910/1000, Train Loss: 0.0017, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_910.pth
Epoch 911/1000, Train Loss: 0.0017, Val Loss: 0.0016
Epoch 912/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 913/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 914/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 915/1000, Train Loss: 0.0016, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_915.pth
Epoch 916/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 917/1000, Train Loss: 0.0016, Val Loss: 0.0017
Epoch 918/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 919/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 920/1000, Train Loss: 0.0016, Val Loss: 0.0017
模型已保存至 ./checkpoints/checkpoint_epoch_920.pth
Epoch 921/1000, Train Loss: 0.0015, Val Loss: 0.0017
Epoch 922/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 923/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 924/1000, Train Loss: 0.0015, Val Loss: 0.0017
Epoch 925/1000, Train Loss: 0.0015, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_925.pth
Epoch 926/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 927/1000, Train Loss: 0.0015, Val Loss: 0.0017
Epoch 928/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 929/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 930/1000, Train Loss: 0.0015, Val Loss: 0.0017
模型已保存至 ./checkpoints/checkpoint_epoch_930.pth
Epoch 931/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 932/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 933/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 934/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 935/1000, Train Loss: 0.0015, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_935.pth
Epoch 936/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 937/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 938/1000, Train Loss: 0.0014, Val Loss: 0.0016
Epoch 939/1000, Train Loss: 0.0014, Val Loss: 0.0016
Epoch 940/1000, Train Loss: 0.0014, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_940.pth
Epoch 941/1000, Train Loss: 0.0014, Val Loss: 0.0016
Epoch 942/1000, Train Loss: 0.0014, Val Loss: 0.0016
Epoch 943/1000, Train Loss: 0.0014, Val Loss: 0.0016
Epoch 944/1000, Train Loss: 0.0014, Val Loss: 0.0015
Epoch 945/1000, Train Loss: 0.0014, Val Loss: 0.0015
模型已保存至 ./checkpoints/checkpoint_epoch_945.pth
Epoch 946/1000, Train Loss: 0.0014, Val Loss: 0.0015
Epoch 947/1000, Train Loss: 0.0014, Val Loss: 0.0016
Epoch 948/1000, Train Loss: 0.0015, Val Loss: 0.0016
Epoch 949/1000, Train Loss: 0.0015, Val Loss: 0.0017
Epoch 950/1000, Train Loss: 0.0016, Val Loss: 0.0017
模型已保存至 ./checkpoints/checkpoint_epoch_950.pth
Epoch 951/1000, Train Loss: 0.0017, Val Loss: 0.0018
Epoch 952/1000, Train Loss: 0.0017, Val Loss: 0.0014
Epoch 953/1000, Train Loss: 0.0015, Val Loss: 0.0015
Epoch 954/1000, Train Loss: 0.0016, Val Loss: 0.0016
Epoch 955/1000, Train Loss: 0.0016, Val Loss: 0.0015
模型已保存至 ./checkpoints/checkpoint_epoch_955.pth
Epoch 956/1000, Train Loss: 0.0016, Val Loss: 0.0017
Epoch 957/1000, Train Loss: 0.0019, Val Loss: 0.0019
Epoch 958/1000, Train Loss: 0.0020, Val Loss: 0.0016
Epoch 959/1000, Train Loss: 0.0018, Val Loss: 0.0016
Epoch 960/1000, Train Loss: 0.0017, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_960.pth
Epoch 961/1000, Train Loss: 0.0017, Val Loss: 0.0014
Epoch 962/1000, Train Loss: 0.0017, Val Loss: 0.0018
Epoch 963/1000, Train Loss: 0.0017, Val Loss: 0.0015
Epoch 964/1000, Train Loss: 0.0016, Val Loss: 0.0015
Epoch 965/1000, Train Loss: 0.0017, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_965.pth
Epoch 966/1000, Train Loss: 0.0016, Val Loss: 0.0015
Epoch 967/1000, Train Loss: 0.0016, Val Loss: 0.0014
Epoch 968/1000, Train Loss: 0.0016, Val Loss: 0.0015
Epoch 969/1000, Train Loss: 0.0015, Val Loss: 0.0015
Epoch 970/1000, Train Loss: 0.0015, Val Loss: 0.0016
模型已保存至 ./checkpoints/checkpoint_epoch_970.pth
Epoch 971/1000, Train Loss: 0.0015, Val Loss: 0.0015
Epoch 972/1000, Train Loss: 0.0015, Val Loss: 0.0015
Epoch 973/1000, Train Loss: 0.0015, Val Loss: 0.0015
Epoch 974/1000, Train Loss: 0.0015, Val Loss: 0.0015
Epoch 975/1000, Train Loss: 0.0014, Val Loss: 0.0015
模型已保存至 ./checkpoints/checkpoint_epoch_975.pth
Epoch 976/1000, Train Loss: 0.0014, Val Loss: 0.0015
Epoch 977/1000, Train Loss: 0.0014, Val Loss: 0.0015
Epoch 978/1000, Train Loss: 0.0014, Val Loss: 0.0015
Epoch 979/1000, Train Loss: 0.0014, Val Loss: 0.0015
Epoch 980/1000, Train Loss: 0.0014, Val Loss: 0.0015
模型已保存至 ./checkpoints/checkpoint_epoch_980.pth
Epoch 981/1000, Train Loss: 0.0014, Val Loss: 0.0014
Epoch 982/1000, Train Loss: 0.0014, Val Loss: 0.0014
Epoch 983/1000, Train Loss: 0.0014, Val Loss: 0.0014
Epoch 984/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 985/1000, Train Loss: 0.0013, Val Loss: 0.0014
模型已保存至 ./checkpoints/checkpoint_epoch_985.pth
Epoch 986/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 987/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 988/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 989/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 990/1000, Train Loss: 0.0013, Val Loss: 0.0014
模型已保存至 ./checkpoints/checkpoint_epoch_990.pth
Epoch 991/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 992/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 993/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 994/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 995/1000, Train Loss: 0.0013, Val Loss: 0.0014
模型已保存至 ./checkpoints/checkpoint_epoch_995.pth
Epoch 996/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 997/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 998/1000, Train Loss: 0.0013, Val Loss: 0.0014
Epoch 999/1000, Train Loss: 0.0012, Val Loss: 0.0014
Epoch 1000/1000, Train Loss: 0.0012, Val Loss: 0.0013
模型已保存至 ./checkpoints/checkpoint_epoch_1000.pth


    '''