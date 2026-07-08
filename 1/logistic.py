import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# 准确率评估函数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        logits = net(X)
        # 二分类：大于0判1，否则0
        pred = (logits > 0).squeeze().float()
        acc_sum += (pred == y).sum().item()
        n += y.size(0)
    return acc_sum / n if n > 0 else 0


# 手动SGD优化
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size


# 批量数据迭代器
def data_iter(batch_size, x, labels):
    num_examples = len(x)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i : min(i + batch_size, num_examples)])
        yield x.index_select(0, j), labels.index_select(0, j)


if __name__ == "__main__":
    # 构建二分类数据集
    n_data = torch.ones(50, 2)
    x1 = torch.normal(2 * n_data, 1)
    y1 = torch.zeros(50)
    x2 = torch.normal(-2 * n_data, 1)
    y2 = torch.ones(50)

    x = torch.cat((x1, x2), 0).float()
    y = torch.cat((y1, y2), 0).float()  # BCEWithLogitsLoss标签需float类型

    # 新建输出文件夹
    os.makedirs("./output", exist_ok=True)

    # 绘制原始数据集
    plt.scatter(
        x.detach().numpy()[:, 0],
        x.detach().numpy()[:, 1],
        c=y.detach().numpy(),
        s=100,
        lw=0,
        cmap="RdYlGn",
    )
    plt.savefig("./output/datasets.png", bbox_inches="tight")
    plt.close()

    # ====================== 单层逻辑回归 核心 ======================
    # 输入2维，输出1个logit，仅一层权重偏置
    W = torch.tensor(
        np.random.normal(0, 0.01, (2, 1)), dtype=torch.float32, requires_grad=True
    )
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def net(X):
        return X @ W + b  # 单层线性，无激活，直接输出logit

    # 官方损失：内置Sigmoid + BCELoss，无需手动写激活与loss公式
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 超参数
    lr = 0.03
    num_epochs = 20
    batch_size = 10
    params = [W, b]

    loss_list = []
    acc_list = []
    epoch_list = list(range(1, num_epochs + 1))

    for epoch in range(num_epochs):
        for X_batch, y_batch in data_iter(batch_size, x, y):
            pred = net(X_batch)
            # 直接喂入原始logit与浮点标签
            loss = loss_fn(pred, y_batch.unsqueeze(1))
            loss.backward()
            sgd(params, lr, batch_size)
            # 梯度清零
            for p in params:
                p.grad.zero_()

        # 全集损失与精度
        total_pred = net(x)
        total_loss = loss_fn(total_pred, y.unsqueeze(1))
        current_loss = total_loss.item()
        acc = evaluate_accuracy(data_iter(batch_size, x, y), net)

        loss_list.append(current_loss)
        acc_list.append(acc)
        print(f"epoch {epoch + 1:2d} | loss: {current_loss:.4f} | acc: {acc:.4f}")

    # 绘制Loss & Acc 双轴曲线
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epoch_list, loss_list, color="tab:blue", label="Train Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:orange")
    ax2.plot(
        epoch_list, acc_list, color="tab:orange", linestyle="--", label="Train Acc"
    )
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Loss & Accuracy Curve | Single Layer Logistic + BCEWithLogitsLoss")
    plt.tight_layout()
    plt.savefig("./output/loss_acc_curve.png", dpi=200, bbox_inches="tight")
    plt.show()


"""
python 1/logistic.py
epoch  1 | loss: 0.6448 | acc: 0.9900
epoch  2 | loss: 0.5931 | acc: 1.0000
epoch  3 | loss: 0.5480 | acc: 1.0000
epoch  4 | loss: 0.5083 | acc: 1.0000
epoch  5 | loss: 0.4735 | acc: 1.0000
epoch  6 | loss: 0.4428 | acc: 1.0000
epoch  7 | loss: 0.4155 | acc: 1.0000
epoch  8 | loss: 0.3913 | acc: 1.0000
epoch  9 | loss: 0.3696 | acc: 1.0000
epoch 10 | loss: 0.3501 | acc: 1.0000
epoch 11 | loss: 0.3326 | acc: 1.0000
epoch 12 | loss: 0.3167 | acc: 1.0000
epoch 13 | loss: 0.3023 | acc: 1.0000
epoch 14 | loss: 0.2892 | acc: 1.0000
epoch 15 | loss: 0.2772 | acc: 1.0000
epoch 16 | loss: 0.2662 | acc: 1.0000
epoch 17 | loss: 0.2561 | acc: 1.0000
epoch 18 | loss: 0.2467 | acc: 1.0000
epoch 19 | loss: 0.2380 | acc: 1.0000
epoch 20 | loss: 0.2300 | acc: 1.0000
"""