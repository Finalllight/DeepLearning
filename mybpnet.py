import torch
from torch import nn, reshape
import numpy as np
import time
import os

from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_mnist(path, kind='train'):
    """加载MNIST数据从原始文件"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels
# 训练集文件
data_path = 'data/MNIST/raw'

train_images, train_labels = load_mnist(data_path, kind='train')
test_images, test_labels = load_mnist(data_path, kind='t10k')

# 数据预处理
# 归一化到[0,1]范围
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = torch.tensor(train_images).to(device)
train_labels = torch.tensor(train_labels).to(device)
test_images = torch.tensor(test_images).to(device)
test_labels = torch.tensor(test_labels).to(device)

batch_size = 128
num_workers = 0 if os.name == 'nt' else 4  # 根据系统自动设置多进程数量

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False  # 对GPU训练更友好
)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False
)


class mybpnet(nn.Module):
    def __init__(self):

        super(mybpnet,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 输出层不使用softmax，交给损失函数处理
        )
    def forward(self,x):
        return self.layers(x)


model = mybpnet().to(device)
epochs=15

criterion = nn.CrossEntropyLoss()  # 内置softmax，无需在模型中单独设置
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)


for round in range(epochs):
    running_loss = 0
    model.train()  # 切换到训练模式
    for images, labels in train_loader:
        # 清空梯度
        optimizer.zero_grad()

        output=model(images)
        loss = criterion(output, labels)
        # 进行反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

        # 累加损失
        running_loss += loss.item()
    print("epoch{} - Training loss =  {}".format(round, running_loss / batch_size))