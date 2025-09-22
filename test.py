import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
plt.rc("font", family='Microsoft YaHei')
# 目标函数: y = 2x³ + 5x² + 0.7x + 8
torch.manual_seed(42)
np.random.seed(42)


def generate_data(n_samples=100, noise=5.0):
    """生成带噪声的三次多项式数据"""
    x = np.linspace(-5, 3, n_samples)  # 选择适合三次函数的区间
    y_true = 2 * (x ** 3) + 5 * (x ** 2) + 0.7 * x + 8  # 真实函数
    y = y_true + noise * np.random.randn(n_samples)  # 添加高斯噪声
    return x, y, y_true


# 生成数据
x, y, y_true = generate_data()

# 多项式特征转换（三次多项式）
poly = PolynomialFeatures(degree=3, include_bias=False)  # 生成[x, x², x³]
x_poly = poly.fit_transform(x.reshape(-1, 1))  # 转换为三维特征

# 转换为PyTorch张量
x_tensor = torch.from_numpy(x_poly).float()
y_tensor = torch.from_numpy(y).float().view(-1, 1)  # 调整为列向量


# 定义多项式回归模型
class PolynomialRegression(nn.Module):
    def __init__(self, input_size):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层：输入维度=3，输出维度=1

    def forward(self, x):
        return self.linear(x)  # 前向传播


# 初始化模型、损失函数和优化器
model = PolynomialRegression(input_size=x_poly.shape[1])  # 输入维度=3（x, x², x³）
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # 较小的学习率（避免梯度爆炸）
num_epochs = 300000  # 三次多项式需要更多训练轮次
losses = []

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    predictions = model(x_tensor)

    # 计算损失
    loss = criterion(predictions, y_tensor)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每30000轮打印一次损失
    if (epoch + 1) % 30000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 提取模型参数
weights = model.linear.weight.data.numpy()[0]  # [w1, w2, w3] 对应x, x², x³的系数
bias = model.linear.bias.data.numpy()[0]  # 常数项b

# 打印参数对比
print(f'\nLearned parameters:')
print(f'w1 (x系数) = {weights[0]:.4f} (预期0.7)')
print(f'w2 (x²系数) = {weights[1]:.4f} (预期5)')
print(f'w3 (x³系数) = {weights[2]:.4f} (预期2)')
print(f'b (常数项) = {bias:.4f} (预期8)')

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='带噪声的数据')
plt.plot(x, y_true, 'r--', label='真实曲线 y=2x³+5x²+0.7x+8')
plt.plot(x, model(x_tensor).detach().numpy(), 'g-', linewidth=2, label='拟合曲线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'三次多项式回归 (MSE={loss.item():.4f})')
plt.show()
