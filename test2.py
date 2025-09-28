import torch
from torch import nn
import numpy as np
import time
import os
import random
from tqdm import tqdm  # 进度条库，提升可视化体验
from torch.utils.data import TensorDataset, DataLoader


# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()

# 设备配置，兼容CPU和GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_mnist(path, kind='train'):
    """加载MNIST数据从原始文件，添加错误处理"""
    try:
        labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
        images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

        # 检查文件是否存在
        if not os.path.exists(labels_path) or not os.path.exists(images_path):
            raise FileNotFoundError(f"MNIST文件不存在于路径: {path}")

        with open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with open(images_path, 'rb') as imgpath:
            # 计算图像数量并reshape，避免硬编码
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = images.reshape(len(labels), 784)  # 28x28=784

        return images, labels

    except Exception as e:
        print(f"加载数据出错: {e}")
        raise  # 抛出异常，避免程序继续运行


# 数据加载与预处理
data_path = 'data/MNIST/raw'
train_images, train_labels = load_mnist(data_path, kind='train')
test_images, test_labels = load_mnist(data_path, kind='t10k')

# 数据预处理：归一化并转换为tensor，直接在转换时指定设备，减少数据迁移开销
train_images = torch.tensor(train_images.astype(np.float32) / 255.0, device=device)
train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)  # 标签需为long类型
test_images = torch.tensor(test_images.astype(np.float32) / 255.0, device=device)
test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)

# 数据加载器，设置num_workers加速数据加载（Windows下建议设为0）
batch_size = 128
num_workers = 0 if os.name == 'nt' else 4  # 根据系统自动设置多进程数量

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True  # 对GPU训练更友好
)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


class MyBPNet(nn.Module):
    """优化后的BP网络，代码更简洁，移除冗余操作"""

    def __init__(self):
        super(MyBPNet, self).__init__()
        # 用Sequential整合网络层，结构更清晰
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 输出层不使用softmax，交给损失函数处理
        )

    def forward(self, x):
        # 输入已经是展平的784维向量，无需额外展平操作
        return self.layers(x)


# 初始化模型、损失函数和优化器
model = MyBPNet().to(device)  # 用to(device)替代.cuda()，兼容CPU
criterion = nn.CrossEntropyLoss()  # 内置softmax，无需在模型中单独设置
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

# 记录最佳测试准确率，用于保存模型
best_acc = 0.0
epochs = 15
save_path = "best_bp_model.pth"  # 最佳模型保存路径


def evaluate(model, test_loader, criterion):
    """评估函数，计算测试集损失和准确率"""
    model.eval()  # 切换到评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    acc = 100 * correct / total
    return avg_loss, acc


# 训练循环
for epoch in range(epochs):
    start_time = time.time()  # 记录每个epoch的开始时间
    model.train()  # 切换到训练模式
    running_loss = 0.0

    # 使用tqdm添加进度条，直观显示训练进度
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for images, labels in train_bar:
        optimizer.zero_grad()  # 清空梯度

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        # 在进度条上实时显示当前批次损失
        train_bar.set_postfix(loss=loss.item())

    # 计算训练集平均损失
    avg_train_loss = running_loss / len(train_loader)

    # 评估测试集
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    # 计算epoch耗时
    epoch_time = time.time() - start_time

    # 打印训练信息
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print(f"训练损失: {avg_train_loss:.4f} | 测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}% | 耗时: {epoch_time:.2f}秒")

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), save_path)
        print(f"保存最佳模型 (准确率: {best_acc:.2f}%) 到 {save_path}")

print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
