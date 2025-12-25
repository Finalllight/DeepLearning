# 第 1 步：构建 BP 网络模型
import torch
import torchvision
from torchvision import transforms


# MNIST 包含 70,000 张手写数字图像: 60,000 张用于训练，10,000 张用于测试。
# 图像是灰度的，28×28 像素的，并且居中的，以减少预处理和加快运行。
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
print("reading data\n")
# 使用 torchvision 读取数据
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print("loading data\n")
# 使用 DataLoader 加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
class BPNetwork(torch.nn.Module):

    def __init__(self):
        super(BPNetwork, self).__init__()

        """
        定义第一个线性层，
        输入为图片（28x28），
        输出为第一个隐层的输入，大小为 128。
        """
        self.linear1 = torch.nn.Linear(28 * 28, 128)
        # 在第一个隐层使用 ReLU 激活函数
        self.relu1 = torch.nn.ReLU()
        """
        定义第二个线性层，
        输入是第一个隐层的输出，
        输出为第二个隐层的输入，大小为 64。
        """
        self.linear2 = torch.nn.Linear(128, 64)
        # 在第二个隐层使用 ReLU 激活函数
        self.relu2 = torch.nn.ReLU()
        """
        定义第三个线性层，
        输入是第二个隐层的输出，
        输出为输出层，大小为 10
        """
        self.linear3 = torch.nn.Linear(64, 10)
        # 最终的输出经过 softmax 进行归一化
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        定义神经网络的前向传播
        x: 图片数据, shape 为(64, 1, 28, 28)
        """
        # 首先将 x 的 shape 转为(64, 784)
        x = x.view(x.shape[0], -1)

        # 接下来进行前向传播
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.softmax(x)

        # 上述一串，可以直接使用 x = self.model(x) 代替。

        return x

print("preparing model\n")
model = BPNetwork()
# criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()  # 定义 loss 函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)  # 定义优化器
print("start training\n")
epochs = 15  # 一共训练 15 轮
for i in range(epochs):
    running_loss = 0  # 本轮的损失值
    for images, labels in trainloader:
        # 前向传播获取预测值
        output = model(images)
        # 计算损失
        loss = criterion(output, labels)
        # 进行反向传播
        loss.backward()
        # 更新权重
        optimizer.step()
        # 清空梯度
        optimizer.zero_grad()
        # 累加损失
        running_loss += loss.item()

    # 一轮循环结束后打印本轮的损失函数
    print("Epoch {} - Training loss: {}".format(i, running_loss / len(trainloader)))
