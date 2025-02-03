import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib
import japanize_matplotlib
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

# ハイパーパラメータ設定
batch_size = 128
num_classes = 10
epochs = 50

# 画像の入力サイズ
img_rows, img_cols = 28, 28

# データ読み込み関数
def load(f):
    return np.load(f)['arr_0']

# データのロード
x_train = load("drive/My Drive/zemi/kmnist-train-imgs.npz")
x_test = load("drive/My Drive/zemi/kmnist-test-imgs.npz")
y_train = load("drive/My Drive/zemi/kmnist-train-labels.npz")
y_test = load("drive/My Drive/zemi/kmnist-test-labels.npz")

# データの形状を調整
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# データの正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

# クラスラベルをone-hotエンコード
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# モデル構築
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# モデルコンパイル
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# 学習率スケジューラー
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.8, min_lr=0.00001)

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    return lr * 0.95

scheduler = LearningRateScheduler(lr_scheduler)

# モデル学習
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    epochs=epochs,
    steps_per_epoch=len(x_train) // batch_size,
    callbacks=[lr_reduction, scheduler]
)


# モデルの評価
train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

# KMNISTのクラスラベルに対応する日本語文字
class_names = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']

# 予測と混同行列の生成
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 予測値
y_true = np.argmax(y_test, axis=1)  # 実際のラベル

# 混同行列を生成
conf_matrix = confusion_matrix(y_true, y_pred_classes)

report = classification_report(y_true, y_pred_classes, target_names=class_names, digits=4)
print("Classification Report:\n")
print(report)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

errors = (y_pred_classes != y_true)
X_val_errors = x_test[errors]
Y_pred_classes_errors = y_pred_classes[errors]
Y_true_errors = y_true[errors]
Y_pred_errors = y_pred[errors]

Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

sorted_delta_errors = np.argsort(delta_pred_true_errors)
sorted_delta_errors_inv = np.flipud(sorted_delta_errors)

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    n = 0
    nrows = 3
    ncols = 6
    num_errors_to_display = min(len(errors_index), nrows * ncols)

    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12, 6))

    for row in range(nrows):
        for col in range(ncols):
            if n < num_errors_to_display:
                error = errors_index[n]
                pred_class_index = pred_errors[error]
                true_class_index = obs_errors[error]
                ax[row, col].imshow(img_errors[error].reshape((28, 28)), cmap='gray')
                ax[row, col].set_title(f"P:{class_names[pred_errors[error]]} T:{class_names[obs_errors[error]]}")
                ax[row, col].axis('off')
                n += 1
            else:
                ax[row, col].axis('off')
    plt.tight_layout()
    plt.show()

display_errors(sorted_delta_errors_inv[:6], X_val_errors, Y_pred_classes_errors, Y_true_errors)

def plot_accuracy_loss(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_accuracy_loss(history)

