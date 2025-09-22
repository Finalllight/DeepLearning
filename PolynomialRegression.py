import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)


# 1. 生成模拟数据（非线性关系）
def generate_data(n_samples=100, noise=0.2):
    # 生成x值
    x = np.linspace(-3, 3, n_samples)

    # 真实的二次函数关系：y = 0.5x^2 + x + 2
    y_true = 0.5 * x ** 2 + x + 2

    # 添加噪声
    y = y_true + noise * np.random.randn(n_samples)

    return x, y, y_true


# 生成数据
x, y, y_true = generate_data()

# 可视化原始数据


# 2. 准备数据 - 创建多项式特征
# 使用Scikit-learn的PolynomialFeatures来创建多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x.reshape(-1, 1))

# 转换为PyTorch张量
x_tensor = torch.from_numpy(x_poly).float()
y_tensor = torch.from_numpy(y).float().view(-1, 1)


# 3. 定义多项式回归模型
class PolynomialRegression(nn.Module):
    def __init__(self, input_size):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# 创建模型实例
# 输入大小是多项式的特征数（对于degree=2，有x和x^2两个特征）
model = PolynomialRegression(input_size=x_poly.shape[1])

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 5. 训练模型
num_epochs = 1000
losses = []

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

    # 每100轮打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# # 6. 评估模型
# model.eval()
# with torch.no_grad():
#     predictions = model(x_tensor)
#     mse = mean_squared_error(y, predictions.numpy())
#     print(f'\nFinal Mean Squared Error: {mse:.4f}')
#
# 获取训练后的参数
weights = model.linear.weight.data.numpy()[0]
bias = model.linear.bias.data.numpy()[0]
print(f'Learned parameters: w1 = {weights[0]:.4f}, w2 = {weights[1]:.4f}, b = {bias:.4f}')
print(f'True parameters: w1 = 1.0, w2 = 0.5, b = 2.0')

# # 7. 可视化结果
# plt.figure(figsize=(15, 5))
#
# # 左图：拟合结果
# plt.subplot(1, 3, 1)
# plt.scatter(x, y, label='Noisy Data', alpha=0.6)
# plt.plot(x, y_true, 'r-', label='True Function', linewidth=2)
# plt.plot(x, predictions.numpy(), 'g--', label='Fitted Curve', linewidth=2)
# plt.title('Polynomial Regression Fit')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
#
# # 中图：损失下降曲线
# plt.subplot(1, 3, 2)
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.yscale('log')
#
# # 右图：不同多项式阶数的比较
# plt.subplot(1, 3, 3)
# plt.scatter(x, y, label='Noisy Data', alpha=0.6)
# plt.plot(x, y_true, 'r-', label='True Function', linewidth=2)
#
# # 尝试不同阶数的多项式
# degrees = [1, 2, 4, 8]
# colors = ['orange', 'green', 'purple', 'brown']
#
# for i, degree in enumerate(degrees):
#     poly = PolynomialFeatures(degree=degree, include_bias=False)
#     x_poly_test = poly.fit_transform(x.reshape(-1, 1))
#     x_tensor_test = torch.from_numpy(x_poly_test).float()
#
#     # 创建并训练新模型
#     model_test = PolynomialRegression(input_size=x_poly_test.shape[1])
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.SGD(model_test.parameters(), lr=0.01)
#
#     # 快速训练（仅用于演示）
#     for epoch in range(500):
#         predictions_test = model_test(x_tensor_test)
#         loss = criterion(predictions_test, y_tensor)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # 绘制拟合曲线
#     with torch.no_grad():
#         predictions_test = model_test(x_tensor_test)
#         plt.plot(x, predictions_test.numpy(),
#                  color=colors[i],
#                  linestyle='--',
#                  label=f'Degree {degree}')
#
# plt.title('Comparison of Different Polynomial Degrees')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 8. 过拟合演示
# # 使用非常高阶的多项式（degree=15）来演示过拟合
# poly_high = PolynomialFeatures(degree=15, include_bias=False)
# x_poly_high = poly_high.fit_transform(x.reshape(-1, 1))
# x_tensor_high = torch.from_numpy(x_poly_high).float()
#
# model_high = PolynomialRegression(input_size=x_poly_high.shape[1])
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model_high.parameters(), lr=0.01)
#
# # 训练高阶多项式模型
# for epoch in range(2000):
#     predictions_high = model_high(x_tensor_high)
#     loss = criterion(predictions_high, y_tensor)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# # 生成测试数据（与训练数据不同的范围）
# x_test = np.linspace(-5, 5, 100)
# y_test_true = 0.5 * x_test ** 2 + x_test + 2
# x_test_poly = poly_high.transform(x_test.reshape(-1, 1))
# x_test_tensor = torch.from_numpy(x_test_poly).float()
#
# # 可视化过拟合
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, label='Training Data', alpha=0.6)
# plt.plot(x_test, y_test_true, 'r-', label='True Function', linewidth=2)
#
# with torch.no_grad():
#     predictions_high_train = model_high(x_tensor_high)
#     predictions_high_test = model_high(x_test_tensor)
#
#     plt.plot(x, predictions_high_train.numpy(), 'g--', label='High Degree Fit (Train)', linewidth=2)
#     plt.plot(x_test, predictions_high_test.numpy(), 'm--', label='High Degree Fit (Test)', linewidth=2)
#
# plt.title('Overfitting with High Degree Polynomial (Degree=15)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
