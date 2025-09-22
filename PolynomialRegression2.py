import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
#y=2*x^3+5*x^2+0.7*x+8
torch.manual_seed(42)
np.random.seed(42)

def generate_data(n_samples=500,noise=0.2):
    x = np.linspace(-8, 8, n_samples)
    y_true = 2 * (x ** 3) + 5 * (x ** 2) + 0.7 * x + 8
    # y_true = 0.5 * x ** 2 + x + 2
    y = y_true +noise * np.random.randn(n_samples)
    return  x,y,y_true
x, y, y_true = generate_data()

poly = PolynomialFeatures(degree=3, include_bias=False)
x_poly = poly.fit_transform(x.reshape(-1, 1))

x_tensor = torch.from_numpy(x_poly).float().cuda()
y_tensor = torch.from_numpy(y).float().view(-1, 1).cuda()

class PolynomialRegression(nn.Module):
    def __init__(self, input_size):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

model = PolynomialRegression(input_size=x_poly.shape[1])
model = model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
num_epochs = 500000
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
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
model=model.cpu()
weights = model.linear.weight.data.numpy()[0]
bias = model.linear.bias.data.numpy()[0]

print(f'Learned parameters:  w0 = {weights[0]:.4f},w1 = {weights[1]:.4f}, w2 = {weights[2]:.4f}, b = {bias:.4f}')
print(f'True parameters: w0=2,w1 = 5, w2 = 0.7, b = 8')
x_tensor=x_tensor.cpu()
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label='noise')
plt.plot(x, y_true, 'r--', label='true y=2x³+5x²+0.7x+8')
plt.plot(x, model(x_tensor).detach().numpy(), 'g-', linewidth=2, label='regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
