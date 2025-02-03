import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data = np.loadtxt("drive/My Drive/3class.txt")
X = data[:, :-1]
y = data[:, -1].astype(int)

def standardize(X):
   mean = np.mean(X, axis=0)
   std = np.std(X, axis=0)
   return (X - mean) / std

X = standardize(X)

def train_test_split(X, y, test=0.3, rand=42):
    np.random.seed(rand)
    indices = np.random.permutation(X.shape[0])
    test = int(X.shape[0] * test)
    train_indices, test_indices = indices[test:], indices[:test]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y)

def OneHot(y, num):
    return np.eye(num)[y]

y_train_one_hot = OneHot(y_train, num=3)
y_test_one_hot = OneHot(y_test, num=3)

class NeuralNetwork:
    def __init__(self, input, hidden, output):
        self.W1 = np.random.randn(input, hidden) * np.sqrt(2 / input)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, output) * np.sqrt(2 / hidden)
        self.b2 = np.zeros((1, output))

        self.mW1, self.mW2, self.mb1, self.mb2 = np.zeros_like(self.W1), np.zeros_like(self.W2), np.zeros_like(self.b1), np.zeros_like(self.b2)
        self.vW1, self.vW2, self.vb1, self.vb2 = np.zeros_like(self.W1), np.zeros_like(self.W2), np.zeros_like(self.b1), np.zeros_like(self.b2)
        self.t = 0

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0.01 * self.Z1, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def backward(self, X, y, rate=0.01):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.A1 > 0)

        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.update(dW1, dW2, db1, db2, rate)

    def update(self, dW1, dW2, db1, db2, rate=0.001, beta1=0.9, beta2=0.999):
        self.t += 1
        self.mW1 = beta1 * self.mW1 + (1 - beta1) * dW1
        self.mW2 = beta1 * self.mW2 + (1 - beta1) * dW2
        self.mb1 = beta1 * self.mb1 + (1 - beta1) * db1
        self.mb2 = beta1 * self.mb2 + (1 - beta1) * db2

        self.vW1 = beta2 * self.vW1 + (1 - beta2) * np.square(dW1)
        self.vW2 = beta2 * self.vW2 + (1 - beta2) * np.square(dW2)
        self.vb1 = beta2 * self.vb1 + (1 - beta2) * np.square(db1)
        self.vb2 = beta2 * self.vb2 + (1 - beta2) * np.square(db2)

        mW1_hat = self.mW1 / (1 - beta1**self.t)
        mW2_hat = self.mW2 / (1 - beta1**self.t)
        mb1_hat = self.mb1 / (1 - beta1**self.t)
        mb2_hat = self.mb2 / (1 - beta1**self.t)

        vW1_hat = self.vW1 / (1 - beta2**self.t)
        vW2_hat = self.vW2 / (1 - beta2**self.t)
        vb1_hat = self.vb1 / (1 - beta2**self.t)
        vb2_hat = self.vb2 / (1 - beta2**self.t)

        self.W1 -= rate * mW1_hat / (np.sqrt(vW1_hat) + 1e-8)
        self.W2 -= rate * mW2_hat / (np.sqrt(vW2_hat) + 1e-8)
        self.b1 -= rate * mb1_hat / (np.sqrt(vb1_hat) + 1e-8)
        self.b2 -= rate * mb2_hat / (np.sqrt(vb2_hat) + 1e-8)

def cross(y_true, y_pred):
    m = y_true.shape[0]
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def train(nn, X_train, y_train, X_test, y_test, epochs=100001, rate=0.001, batch=32):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch):
            X_batch = X_train[i:i+batch]
            y_batch = y_train[i:i+batch]
            output = nn.forward(X_batch)
            nn.backward(X_batch, y_batch, rate)
        train_loss = cross(y_train, nn.forward(X_train))
        train_accuracy = np.mean(np.argmax(nn.forward(X_train), axis=1) == np.argmax(y_train, axis=1))
        test_accuracy = np.mean(np.argmax(nn.forward(X_test), axis=1) == np.argmax(y_test, axis=1))
        test_loss = cross(y_test, nn.forward(X_test))

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}, Test Loss = {test_loss}, Test Accuracy = {test_accuracy}")

    return train_losses, train_accuracies, test_losses, test_accuracies

nn = NeuralNetwork(input=2, hidden=64, output=3)

train_losses, train_accuracies, test_losses, test_accuracies = train(nn, X_train, y_train_one_hot, X_test, y_test_one_hot)

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
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

map = mcolors.ListedColormap(['red', 'lightgreen', 'blue'])

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.75, cmap=map)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=map, edgecolors='k', marker='o', alpha=0.7)
plt.title('3class')
plt.xlabel('x')
plt.ylabel('y')
plt.show()