import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成二分类数据集
X, y = make_classification(
    n_samples=1000,  # 样本数量
    n_features=2,  # 特征数量
    n_informative=2,  # 有效特征数量
    n_redundant=0,  # 冗余特征数量
    random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)  # 转换为列向量
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)


# 2. 定义Logistic回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        # 线性层：输入维度为特征数，输出维度为1（二分类）
        self.linear = nn.Linear(input_dim, 1)
        # Sigmoid函数：将线性输出转换为概率
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 前向传播：线性变换 -> Sigmoid激活
        out = self.linear(x)
        out = self.sigmoid(out)
        return out


# 3. 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]  # 输入维度为特征数量
model = LogisticRegression(input_dim)

# 二分类交叉熵损失（BCEWithLogitsLoss会自动包含Sigmoid，这里我们手动实现了Sigmoid，所以用BCELoss）
criterion = nn.BCELoss()

# 优化器：随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 10000
losses = []

for epoch in range(num_epochs):
    # 前向传播：计算预测概率
    y_pred = model(X_train_tensor)

    # 计算损失
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())

    # 反向传播和参数更新
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每1000轮打印一次损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 模型评估
with torch.no_grad():  # 关闭梯度计算，节省内存
    # 在测试集上预测
    y_pred_prob = model(X_test_tensor)
    # 将概率转换为类别（阈值为0.5）
    y_pred = (y_pred_prob >= 0.5).float()

    # 计算准确率
    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())
    print(f'\ncorrect rate: {accuracy:.4f}')

# 6. 可视化决策边界
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')


# 绘制决策边界
plt.subplot(1, 2, 2)
# 生成网格点
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点类别
with torch.no_grad():
    mesh_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    Z = model(mesh_points)
    Z = (Z >= 0.5).float().numpy().reshape(xx.shape)

# 绘制决策边界和数据点
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolor='k', cmap=plt.cm.Paired)
plt.xlabel('feature1')
plt.ylabel('feature2')


plt.tight_layout()
plt.show()
