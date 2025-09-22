import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# 数据预处理：将图像转为Tensor，并标准化（均值0.1307，标准差0.3081是MNIST的统计值）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 构建数据加载器（批量加载数据，方便训练）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积部分：提取特征
        self.conv_layers = nn.Sequential(
            # 卷积层1：5×5×1→5×5×6，输出尺寸28-5+1=24
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),  # 激活层
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化后尺寸24/2=12
            # 卷积层2：5×5×6→5×5×16，输出尺寸12-5+1=8
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),  # 激活层
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化后尺寸8/2=4
        )
        # 全连接部分：分类
        self.fc_layers = nn.Sequential(
            # 全连接层1：4×4×16=256 → 120
            nn.Linear(in_features=16 * 4 * 4, out_features=120),
            nn.ReLU(),
            # 全连接层2：120 → 84
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            # 输出层：84 → 10（10个类别）
            nn.Linear(in_features=84, out_features=10)
        )

    # 前向传播（输入→卷积→全连接→输出）
    def forward(self, x):
        x = self.conv_layers(x)  # 卷积部分处理：(batch,1,28,28)→(batch,16,4,4)
        x = x.view(-1, 16 * 4 * 4)  # 拉平：(batch,16,4,4)→(batch,256)
        x = self.fc_layers(x)  # 全连接部分处理：(batch,256)→(batch,10)
        return x
# 设备选择：优先用GPU（若有），否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)  # 模型放入设备

# 损失函数：交叉熵损失（适合多分类，已包含Softmax）
criterion = nn.CrossEntropyLoss()
# 优化器：随机梯度下降（SGD），学习率0.01，动量0.9（加速收敛）
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 模型设为训练模式（启用Dropout等，此处无，但习惯保留）
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据放入设备
        data, target = data.to(device), target.to(device)

        # 前向传播：计算预测值
        output = model(data)
        # 计算损失
        loss = criterion(output, target)

        # 反向传播+参数更新
        optimizer.zero_grad()  # 清空上一轮梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 记录损失
        train_loss += loss.item() * data.size(0)

    # 计算本轮平均损失
    train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}], Train Loss: {train_loss:.4f}')


# 测试模型（评估准确率）
def test(model, test_loader, criterion):
    model.eval()  # 模型设为评估模式（禁用Dropout等）
    test_loss = 0.0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            # 计算预测正确的数量（取概率最大的类别）
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算测试集平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset) * 100
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n')


# 开始训练（训练5轮，可调整）
num_epochs = 5
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)