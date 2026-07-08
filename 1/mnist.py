import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

#  自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据预处理与加载
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
    ]
)

# 训练集 & 测试集
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. 定义Softmax回归模型（单层线性：784 → 10）
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [B, 1, 28, 28] → 展平为 [B, 784]
        x = x.view(x.size(0), -1)
        out = self.linear(x)  # 输出logits，CrossEntropyLoss内部自动Softmax
        return out


model = SoftmaxRegression()
#  模型移至GPU/CPU
model = model.to(device)

# 3. 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)


# 4. 准确率计算函数
def calc_accuracy(loader, net):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for img, label in loader:
            #  数据也迁移到对应设备
            img = img.to(device)
            label = label.to(device)
            logits = net(img)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    net.train()
    return correct / total


# 5. 训练过程 & 记录日志
num_epochs = 30
train_loss_list = []
train_acc_list = []
test_acc_list = []
epoch_list = list(range(1, num_epochs + 1))

os.makedirs("./output", exist_ok=True)

for epoch in range(num_epochs):
    total_loss = 0.0
    for images, labels in train_loader:
        #  每一批数据送入GPU
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播+参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    # 本epoch平均训练loss
    avg_train_loss = total_loss / len(train_dataset)
    # 训练集准确率
    train_acc = calc_accuracy(train_loader, model)
    # 测试集准确率
    test_acc = calc_accuracy(test_loader, model)

    train_loss_list.append(avg_train_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(
        f"Epoch {epoch + 1:2d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
    )

# 6. 绘制Loss & 双Acc曲线
fig, ax1 = plt.subplots(figsize=(12, 6))

# 左Y轴：训练Loss
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Loss", color="#1f77b4")
(line_loss,) = ax1.plot(
    epoch_list, train_loss_list, color="#1f77b4", label="Train Loss"
)
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(alpha=0.3)

# 右Y轴：准确率
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy", color="#ff7f0e")
(line_train_acc,) = ax2.plot(
    epoch_list, train_acc_list, color="#ff7f0e", linestyle="-", label="Train Acc"
)
(line_test_acc,) = ax2.plot(
    epoch_list, test_acc_list, color="#2ca02c", linestyle="--", label="Test Acc"
)
ax2.tick_params(axis="y", labelcolor="#ff7f0e")

# 合并图例
lines = [line_loss, line_train_acc, line_test_acc]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.title("Softmax Regression on Fashion-MNIST | Loss & Accuracy Curve")
plt.tight_layout()
plt.savefig("./output/fmnist_softmax_curve.png", dpi=200, bbox_inches="tight")
plt.show()



"""
python 1/mnist.py
Using device: cuda
Epoch  1 | Train Loss: 0.7172 | Train Acc: 0.8049 | Test Acc: 0.7958
Epoch  2 | Train Loss: 0.5364 | Train Acc: 0.8242 | Test Acc: 0.8109
Epoch  3 | Train Loss: 0.5006 | Train Acc: 0.8300 | Test Acc: 0.8161
Epoch  4 | Train Loss: 0.4813 | Train Acc: 0.8354 | Test Acc: 0.8195
Epoch  5 | Train Loss: 0.4678 | Train Acc: 0.8438 | Test Acc: 0.8275
Epoch  6 | Train Loss: 0.4588 | Train Acc: 0.8408 | Test Acc: 0.8274
Epoch  7 | Train Loss: 0.4515 | Train Acc: 0.8481 | Test Acc: 0.8314
Epoch  8 | Train Loss: 0.4455 | Train Acc: 0.8496 | Test Acc: 0.8316
Epoch  9 | Train Loss: 0.4406 | Train Acc: 0.8495 | Test Acc: 0.8326
Epoch 10 | Train Loss: 0.4369 | Train Acc: 0.8505 | Test Acc: 0.8329
Epoch 11 | Train Loss: 0.4332 | Train Acc: 0.8530 | Test Acc: 0.8354
Epoch 12 | Train Loss: 0.4298 | Train Acc: 0.8531 | Test Acc: 0.8341
Epoch 13 | Train Loss: 0.4277 | Train Acc: 0.8533 | Test Acc: 0.8358
Epoch 14 | Train Loss: 0.4243 | Train Acc: 0.8535 | Test Acc: 0.8355
Epoch 15 | Train Loss: 0.4227 | Train Acc: 0.8567 | Test Acc: 0.8376
Epoch 16 | Train Loss: 0.4207 | Train Acc: 0.8562 | Test Acc: 0.8345
Epoch 17 | Train Loss: 0.4187 | Train Acc: 0.8562 | Test Acc: 0.8364
Epoch 18 | Train Loss: 0.4169 | Train Acc: 0.8548 | Test Acc: 0.8338
Epoch 19 | Train Loss: 0.4148 | Train Acc: 0.8583 | Test Acc: 0.8399
Epoch 20 | Train Loss: 0.4134 | Train Acc: 0.8567 | Test Acc: 0.8373
Epoch 21 | Train Loss: 0.4124 | Train Acc: 0.8540 | Test Acc: 0.8337
Epoch 22 | Train Loss: 0.4105 | Train Acc: 0.8572 | Test Acc: 0.8350
Epoch 23 | Train Loss: 0.4095 | Train Acc: 0.8579 | Test Acc: 0.8403
Epoch 24 | Train Loss: 0.4085 | Train Acc: 0.8562 | Test Acc: 0.8346
Epoch 25 | Train Loss: 0.4076 | Train Acc: 0.8605 | Test Acc: 0.8390
Epoch 26 | Train Loss: 0.4059 | Train Acc: 0.8598 | Test Acc: 0.8410
Epoch 27 | Train Loss: 0.4046 | Train Acc: 0.8609 | Test Acc: 0.8419
Epoch 28 | Train Loss: 0.4041 | Train Acc: 0.8615 | Test Acc: 0.8421
Epoch 29 | Train Loss: 0.4035 | Train Acc: 0.8553 | Test Acc: 0.8384
Epoch 30 | Train Loss: 0.4024 | Train Acc: 0.8612 | Test Acc: 0.8411
"""