import numpy as np


class LogisticRegression:  # stochastic
    def __init__(self):
        self.w = None
        self.b = None

    def forpass(self, x):
        y_hat = np.dot(x, self.w) + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = err * x
        b_grad = 1 * err
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for _ in range(epochs):
            for x_i, y_i in zip(x, y):
                z = self.forpass(x_i)
                y_hat = self.activation(z)
                err = y_hat - y_i
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(z)
        return a > 0.5


class BatchLogisticRegression:  # Batch
    def __init__(self, learning_rate=0.1):
        self.w = None
        self.b = None
        self.lr = learning_rate

    def forpass(self, x):
        y_hat = np.dot(x, self.w) + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = np.dot(np.transpose(x), err)
        b_grad = np.sum(err)
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, epochs=1000):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for _ in range(epochs):
            z = self.forpass(x)
            y_hat = self.activation(z)
            err = y_hat - y
            w_grad, b_grad = self.backprop(x, err)
            self.w -= self.lr * w_grad / x.shape[0]
            self.b -= self.lr * b_grad / x.shape[0]

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(z)
        return a > 0.5
