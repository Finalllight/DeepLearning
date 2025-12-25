import torch
from torch import nn
import numpy as np
import time
import os
from torch.utils.data import DataLoader, TensorDataset

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


# 加载训练集和测试集
data_path = 'data/MNIST/raw'
train_images, train_labels = load_mnist(data_path, kind='train')
test_images, test_labels = load_mnist(data_path, kind='t10k')

# 数据预处理
# 归一化到[0,1]范围
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# 转换为PyTorch张量并移动到设备
train_images = torch.tensor(train_images).to(device)
train_labels = torch.tensor(train_labels).to(device)
test_images = torch.tensor(test_images).to(device)
test_labels = torch.tensor(test_labels).to(device)

print(f"训练集大小: {train_images.shape}")
print(f"测试集大小: {test_images.shape}")

# 创建数据加载器
batch_size = 128

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义BP神经网络模型
class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size1=512, hidden_size2=256, output_size=10):
        super(BPNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        # 激活函数和Dropout防止过拟合
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# 初始化模型
model = BPNeuralNetwork().to(device)
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# 训练函数
def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


# 测试函数
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


# 训练模型
num_epochs = 16
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("开始训练...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    epoch_start = time.time()

    # 训练
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch)

    # 测试
    test_loss, test_acc = test_model(model, test_loader, criterion)

    # 学习率调整
    scheduler.step()

    # 记录结果
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    epoch_time = time.time() - epoch_start
    print(f'Epoch {epoch}/{num_epochs} - '
          f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}% - '
          f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}% - '
          f'时间: {epoch_time:.2f}s')

total_time = time.time() - start_time
print(f'训练完成! 总时间: {total_time:.2f}s')

# 最终评估
final_test_loss, final_test_acc = test_model(model, test_loader, criterion)
print(f'\n最终测试结果:')
print(f'测试损失: {final_test_loss:.4f}')
print(f'测试准确率: {final_test_acc:.2f}%')

# 保存模型
# torch.save(model.state_dict(), 'mnist_bp_model.pth')
# print("模型已保存为 'mnist_bp_model.pth'")
#
# # 可视化训练过程
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 2, 1)
# plt.plot(range(1, num_epochs + 1), train_losses, label='训练损失')
# plt.plot(range(1, num_epochs + 1), test_losses, label='测试损失')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('训练和测试损失')
#
# plt.subplot(1, 2, 2)
# plt.plot(range(1, num_epochs + 1), train_accuracies, label='训练准确率')
# plt.plot(range(1, num_epochs + 1), test_accuracies, label='测试准确率')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.title('训练和测试准确率')
#
# plt.tight_layout()
# plt.savefig('training_curve.png')
# plt.show()
#
#
# # 在单个图像上测试
# def predict_single_image(model, image, true_label):
#     model.eval()
#     with torch.no_grad():
#         image = image.unsqueeze(0)  # 添加批次维度
#         output = model(image)
#         _, predicted = torch.max(output, 1)
#
#         # 可视化图像
#         plt.imshow(image.cpu().squeeze().reshape(28, 28), cmap='gray')
#         plt.title(f'预测: {predicted.item()}, 真实: {true_label}')
#         plt.axis('off')
#         plt.show()
#
#         return predicted.item()
#
#
# # 随机选择一个测试图像进行预测
# import random
#
# idx = random.randint(0, len(test_images) - 1)
# predicted_label = predict_single_image(model, test_images[idx], test_labels[idx].item())
# print(f"预测结果: {predicted_label}, 真实标签: {test_labels[idx].item()}")