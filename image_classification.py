import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# データ変換（データ拡張を追加）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
    transforms.RandomRotation(15),          # ランダム回転
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差
])

# データセットとデータローダーの作成
dataset = datasets.ImageFolder(root="/mnt/c/Users/sumino/Downloads/Liella", transform=transform)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ResNet-18モデルのロード（事前学習済み）
resnet = models.resnet50(pretrained=True)

# 出力層をカスタマイズ
num_classes = len(dataset.classes)  # クラス数
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# モデルをGPUに移動（可能であれば）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet.to(device)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)



from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# 誤分類画像のカウント用の辞書
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 誤分類画像のカウント用の辞書
misclassified_image_count = defaultdict(int)
misclassified_images = []

# 学習ループ
def train_model(model, train_loader, test_loader, epochs=50):
    true_labels_all = []
    predicted_labels_all = []
    
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

    for epoch in range(epochs):
        # モデルを訓練モードに設定
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # 訓練データの結果
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # テストデータの評価
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                # 誤分類画像のカウント
                for i in range(inputs.size(0)):
                    if labels[i] != predicted[i]:
                        img_idx = i + len(test_loader.dataset)  # 画像のインデックスをユニークにする
                        misclassified_image_count[img_idx] += 1
                        misclassified_images.append((inputs[i], labels[i], predicted[i]))
                        
                # ラベルを全体リストに追加
                true_labels_all.extend(labels.cpu().numpy())
                predicted_labels_all.extend(predicted.cpu().numpy())

        test_loss /= len(test_loader)
        test_accuracy = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies, true_labels_all, predicted_labels_all

# 学習の実行
train_losses, test_losses, train_accuracies, test_accuracies, true_labels_all, predicted_labels_all = train_model(model, train_loader, test_loader, epochs=50)



# 学習曲線のプロット
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("3nen_loss.png")

misclassified_labels = [(true_label.item(), pred_label.item()) for _, true_label, pred_label in misclassified_images]
label_combinations = Counter(misclassified_labels)
most_common_combinations = label_combinations.most_common(8)
print("Most common (True, Predicted) label combinations:")
for combination, count in most_common_combinations:
    true_label, pred_label = combination
    print(f"True: {dataset.classes[true_label]}, Pred: {dataset.classes[pred_label]} - Count: {count}")

# 誤分類画像の回数を表示
print("Most common misclassified images:")
for img_idx, count in sorted(misclassified_image_count.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(f"Image {img_idx} misclassified {count} times.")

# 誤分類画像の可視化（誤分類回数が多いものを表示）
plt.figure(figsize=(12, 9))  # 少し小さめに設定
num_rows = 4  # 4行
num_cols = 4  # 4列
misclassified_images_sorted = sorted(misclassified_image_count.items(), key=lambda item: item[1], reverse=True)
# 上位16枚を表示する際に、インデックスが範囲内であることを確認
for idx, (img_idx, misclass_count) in enumerate(misclassified_images_sorted[:16]):
    if img_idx >= len(misclassified_images):  # インデックスが範囲外でないことを確認
        continue  # 範囲外の場合はスキップ

    img, true_label, pred_label = misclassified_images[img_idx]

    img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # 正規化を逆転
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # img が GPU (cuda:0) にある場合
    img = img * torch.tensor(std).to(img.device) + torch.tensor(mean).to(img.device)
    img = img.clamp(0, 1).cpu().numpy()  # GPUからCPUに移動してからNumPy配列に変換

    # プロット位置の指定
    plt.subplot(num_rows, num_cols, idx + 1)
    plt.imshow(img)
    plt.title(
        f"True: {dataset.classes[true_label]} \nPred: {dataset.classes[pred_label]} \nMisclassified: {misclass_count} ",
        fontsize=10,
    )
    plt.axis('off')

plt.tight_layout()
plt.savefig("3nen_gobunrui.png")

# 混同行列の計算
cm = confusion_matrix(true_labels_all, predicted_labels_all)

# 混同行列を表示
plt.figure(figsize=(11, 10))  # 画像全体のサイズを大きく設定
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues)

# ラベルを斜めにする
plt.xticks(rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("3nen_confusion_matrix.png")


# 誤分類回数と正しく分類された回数
correct_labels = Counter(true_labels_all)
pred_labels = Counter(predicted_labels_all)

correct_count = sum(correct_labels.values())
misclassified_count = sum(1 for true, pred in zip(true_labels_all, predicted_labels_all) if true != pred)

print(f"Correctly classified: {correct_count}")
print(f"Misclassified: {misclassified_count}")

misclassified_labels = [(true_label.item(), pred_label.item()) for _, true_label, pred_label in misclassified_images]
label_combinations = Counter(misclassified_labels)
most_common_combinations = label_combinations.most_common(8)
print("Most common (True, Predicted) label combinations:")
for combination, count in most_common_combinations:
    true_label, pred_label = combination
    print(f"True: {dataset.classes[true_label]}, Pred: {dataset.classes[pred_label]} - Count: {count}")

# 誤分類画像の回数を表示
print("Most common misclassified images:")
for img_idx, count in sorted(misclassified_image_count.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(f"Image {img_idx} misclassified {count} times.")
