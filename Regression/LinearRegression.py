import numpy as np


class LinearRegression:  # SGD - 에포크 당 데이터 개수만큼 업데이트(데이터 한 개씩 업데이트함)
    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    def forpass(self, x):
        y_hat = self.w * x + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err
        b_grad = err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        for _ in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                err = y_hat - y_i
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad

    def predict(self, x):
        return self.forpass(x)


class BatchLinearRegression:  # full batch
    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err
        b_grad = err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        for _ in range(epochs):
            y_hat = self.forpass(x)
            err = y_hat - y
            w_grad, b_grad = self.backprop(x, err)
            self.w -= np.mean(w_grad)
            self.b -= np.mean(b_grad)

    def predict(self, x):
        return self.forpass(x)


class BatchLinearRegression:  # full batch
    def __init__(self):
        self.w = 1.0
        self.b = 1.0

    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err
        b_grad = err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100, batch=10):
        for _ in range(epochs):
            y_hat = self.forpass(x)
            err = y_hat - y
            w_grad, b_grad = self.backprop(x, err)
            self.w -= np.mean(w_grad)
            self.b -= np.mean(b_grad)

    def predict(self, x):
        return self.forpass(x)
