import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

# 设备配置：优先使用GPU，没有则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 数据准备
# 定义数据预处理：转为Tensor并标准化到[-1, 1]（DCGAN推荐的范围）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor，像素值范围[0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1,1]（(x-0.5)/0.5）
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True  # 第一次运行会自动下载数据集
)

# 数据加载器：批量加载数据，打乱顺序
batch_size = 128
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2  # 多线程加载数据
)


# 2. 定义生成器(Generator)
# 输入：随机噪声向量(z)，输出：28x28的手写数字图像
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        # 转置卷积层：从噪声向量逐步"放大"到28x28图像
        self.main = nn.Sequential(
            # 输入： latent_dim x 1 x 1（噪声向量）
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),  # 批量归一化，稳定训练
            nn.ReLU(True),  # 激活函数

            # 输出：512 x 7 x 7
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 输出：256 x 14 x 14
            nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # 输出范围[-1,1]，与数据预处理匹配
            # 最终输出：1 x 28 x 28（MNIST是单通道灰度图）
        )

    def forward(self, x):
        # x形状：(batch_size, latent_dim) → 调整为(batch_size, latent_dim, 1, 1)以适应卷积
        x = x.view(x.size(0), self.latent_dim, 1, 1)
        return self.main(x)


# 3. 定义判别器(Discriminator)
# 输入：28x28图像，输出：图像为真实数据的概率（0-1）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 卷积层：从图像提取特征，逐步缩小尺寸
        self.main = nn.Sequential(
            # 输入：1 x 28 x 28（MNIST图像）
            nn.Conv2d(1, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU避免梯度消失

            # 输出：256 x 14 x 14
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出：512 x 7 x 7
            nn.Conv2d(512, 1, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # 输出概率（0-1）
            # 最终输出：1 x 1 x 1 → 展平后为单个概率值
        )

    def forward(self, x):
        x = self.main(x)
        # 展平为(batch_size, 1)的向量
        return x.view(-1, 1)


# 4. 初始化模型、损失函数和优化器
latent_dim = 100  # 噪声向量维度

# 创建生成器和判别器实例，并移动到设备
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 损失函数：二分类交叉熵（适合真假判断）
criterion = nn.BCELoss()

# 优化器：Adam（DCGAN推荐使用，学习率0.0002，beta1=0.5）
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


# 5. 训练函数
def train(num_epochs=20):
    # 创建保存生成图像的文件夹
    os.makedirs('generated_images', exist_ok=True)

    # 固定噪声：用于观察训练过程中生成图像的变化
    fixed_noise = torch.randn(64, latent_dim, device=device)

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)  # 真实图像移到设备

            # 标签：真实图像为1，假图像为0（加入少量噪声避免标签过硬）
            real_labels = torch.full((batch_size, 1), 0.9, device=device)  # 真实标签用0.9而非1
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # 假标签用0.1而非0

            # ---------------------
            #  训练判别器(Discriminator)
            # ---------------------
            # 1. 用真实图像训练D
            discriminator.zero_grad()  # 清空梯度
            outputs = discriminator(real_images)  # D对真实图像的判断
            d_loss_real = criterion(outputs, real_labels)  # 真实图像的损失
            d_loss_real.backward()  # 反向传播计算梯度
            d_real_acc = outputs.mean().item()  # 真实图像的判断准确率（接近1越好）

            # 2. 用生成器生成的假图像训练D
            noise = torch.randn(batch_size, latent_dim, device=device)  # 随机噪声
            fake_images = generator(noise)  # G生成假图像
            outputs = discriminator(fake_images.detach())  # D对假图像的判断（detach避免G被更新）
            d_loss_fake = criterion(outputs, fake_labels)  # 假图像的损失
            d_loss_fake.backward()  # 反向传播计算梯度
            d_fake_acc = outputs.mean().item()  # 假图像的判断准确率（接近0越好）

            # 3. 总损失和参数更新
            d_loss = d_loss_real + d_loss_fake
            optimizer_D.step()  # 更新D的参数

            # ---------------------
            #  训练生成器(Generator)
            # ---------------------
            generator.zero_grad()  # 清空梯度
            outputs = discriminator(fake_images)  # 用更新后的D判断假图像
            # G的目标是让D认为假图像是真实的，所以标签用1
            g_loss = criterion(outputs, torch.full((batch_size, 1), 0.9, device=device))
            g_loss.backward()  # 反向传播计算梯度
            g_acc = outputs.mean().item()  # G的欺骗成功率（接近1越好）
            optimizer_G.step()  # 更新G的参数

            # 打印训练进度
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                      f'D Real Acc: {d_real_acc:.4f}, D Fake Acc: {d_fake_acc:.4f}, G Acc: {g_acc:.4f}')

        # 每个epoch结束后，用固定噪声生成图像并保存
        with torch.no_grad():  # 不计算梯度，节省内存
            fake_images = generator(fixed_noise).detach().cpu()
        save_image(fake_images, f'generated_images/epoch_{epoch + 1}.png', nrow=8, normalize=True)
        print(f"第{epoch + 1}轮生成图像已保存到generated_images文件夹\n")


# 6. 开始训练（建议至少训练20轮，生成效果会逐渐变好）
if __name__ == '__main__':
    train(num_epochs=20)


# 7. 训练完成后，查看生成效果（可单独运行）
def show_generated_images(epoch=20):
    # 加载训练好的生成器（如果需要）
    # generator.load_state_dict(torch.load('generator.pth'))

    # 生成新图像
    noise = torch.randn(16, latent_dim, device=device)
    with torch.no_grad():
        generated = generator(noise).detach().cpu()

    # 显示图像
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated[i].squeeze(), cmap='gray')  # 去除通道维度，显示灰度图
        plt.axis('off')
    plt.show()

# 训练完成后调用此函数查看结果
# show_generated_images()
