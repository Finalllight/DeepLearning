import torch
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()



# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

plt.rc("font", family='Microsoft YaHei')
# 1. 生成模拟数据
# 我们使用 y = 2x + 1 + 噪声 来生成数据
x = np.linspace(-1, 5, 100000).reshape(-1, 1)  # 100个点，范围[-1, 1]
y = 2 * x  + 5+0.3 * np.random.randn(100000, 1)  # 真实关系加上噪声

# 将数据转换为PyTorch张量
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()


# 2. 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 定义一个线性层，输入和输出特征维度都是1
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)


# 创建模型实例
model = LinearRegression()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 随机梯度下降优化器

# 4. 训练模型
num_epochs = 10000  # 训练轮数
losses = []  # 用于记录每轮的损失值

x_tensor = x_tensor.cuda()
y_tensor = y_tensor.cuda()
model=model.cuda()

for epoch in range(num_epochs):
    # 前向传播：计算预测值
    predictions = model(x_tensor)
    # 计算损失
    loss = criterion(predictions, y_tensor)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度缓冲区
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新参数

    # 每10轮打印一次损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
# 将模型设置为评估模式（虽然对于线性回归影响不大，但这是一个好习惯）
model.eval()

# 获取训练后的参数
w = model.linear.weight.item()  # 斜率
b = model.linear.bias.item()  # 截距
print(f'\n训练后的参数: w = {w:.4f}, b = {b:.4f}')
print(f'真实参数: w = 2.0, b = 1.0')

# 6. 可视化结果
# 绘制原始数据点
# plt.figure(figsize=(12, 5))
#
# # 左图：数据点和拟合直线
# plt.subplot(1, 2, 1)
# plt.scatter(x, y, label='原始数据', color='blue', alpha=0.6)
# plt.plot(x, model(x_tensor).detach().numpy(), label='拟合直线', color='red', linewidth=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('一维线性回归')
# plt.legend()
#
# # 右图：损失下降曲线
# plt.subplot(1, 2, 2)
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('训练损失下降曲线')
# plt.yscale('log')  # 使用对数坐标更清楚地显示损失下降
#
# plt.tight_layout()
# plt.show()

# 7. 预测新数据
# 使用训练好的模型进行预测
new_x = torch.tensor([[0.5], [-0.3], [0.8]]).float().cuda()
predictions = model(new_x)

print("\n预测结果:")
for i in range(len(new_x)):
    print(f"x = {new_x[i].item():.2f}, 预测 y = {predictions[i].item():.2f}, " +
          f"真实 y ≈ {2 * new_x[i].item() + 1:.2f}")
end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))