import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
from sklearn.model_selection import train_test_split
import os

# データの読み込み
data = np.loadtxt("drive/My Drive/3class.txt")
X = data[:, :-1]  # 特徴量
y = data[:, -1].astype(int)  # ラベル

# 標準化関数
def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

X, mean, std = standardize(X)

# 訓練データとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# One-Hotエンコーディング関数
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_one_hot = one_hot(y_train, num_classes=3)
y_test_one_hot = one_hot(y_test, num_classes=3)

# Tensorに変換
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_one_hot, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_one_hot, dtype=torch.float32)

# ニューラルネットワークの定義
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの初期化
input_size = X_train.shape[1]
hidden_size = 64
output_size = 3
model = NeuralNetwork(input_size, hidden_size, output_size)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練関数
def train(model, X_train, y_train, X_test, y_test, epochs=5001, batch_size=32, save_gif=False):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    frames = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            X_batch, y_batch = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, torch.max(y_batch, 1)[1])
            loss.backward()
            optimizer.step()

        # 100エポックごとのログ取得
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_loss = criterion(model(X_train), torch.max(y_train, 1)[1]).item()
                test_loss = criterion(model(X_test), torch.max(y_test, 1)[1]).item()
                train_accuracy = (torch.argmax(model(X_train), dim=1) == torch.argmax(y_train, dim=1)).float().mean().item()
                test_accuracy = (torch.argmax(model(X_test), dim=1) == torch.argmax(y_test, dim=1)).float().mean().item()

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}, Test Loss = {test_loss}, Test Accuracy = {test_accuracy}")

            # 決定境界の描画
            map = mcolors.ListedColormap(['red', 'lightgreen', 'blue'])
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

            Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
            Z = torch.argmax(Z, axis=1).numpy().reshape(xx.shape)

            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=map, alpha=0.75)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=map, edgecolors='k', marker='o', alpha=0.7)
            plt.title(f'Epoch {epoch}')

            if save_gif:
                filename = f"frame_{epoch}.png"
                plt.savefig(filename)
                frames.append(filename)
            plt.close()

    # GIF作成
    if save_gif:
        with imageio.get_writer('training_animation.gif', mode='I', duration=0.5) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)
        print("GIF saved as 'training_animation.gif'")

    return train_losses, test_losses, train_accuracies, test_accuracies

# モデルの訓練とGIFの保存
train_losses, test_losses, train_accuracies, test_accuracies = train(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, save_gif=True)

# 損失と精度のプロット
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 一時的な画像ファイルの削除
for frame in os.listdir():
    if frame.startswith("frame_"):
        os.remove(frame)
