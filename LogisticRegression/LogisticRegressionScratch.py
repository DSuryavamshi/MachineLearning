import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd


class LogisticRegression(object):
    def __init__(self, learning_rate, iterations, x, y):
        self.x = x
        self.y = y
        self.m, self.n = x.shape
        self.w = np.zeros((self.n, 1))
        self.lr = learning_rate
        self.iter = iterations

    def y_hat(self):
        wtx = np.dot(self.x, self.w)
        exp = np.exp(-wtx)
        y_prob = 1 / (1 + exp)
        return y_prob

    def loss(self, ypred):
        checkforzero = ypred # To handle log10(0) issue
        checkforzero[checkforzero == 0] = 10 ** -10
        hx = np.where(checkforzero > 10e-10, np.log10(checkforzero), -10)
        hx[hx == -10] = 0

        checkforzero = 1 - ypred # To handle log10(0) issue
        checkforzero[checkforzero == 0] = 10 ** -10
        h_1x = np.where(checkforzero > 10e-10, np.log10(checkforzero), -10)
        loss = -(1 / self.m) * np.sum((self.y * hx) + ((1 - self.y) * h_1x))
        return loss

    def gradient_descent(self):
        for i in range(1, self.iter):
            ypred = self.y_hat()
            final_y = ypred
            loss = self.loss(ypred=ypred)
            self.w = self.w - ((self.lr / self.m) * np.sum((ypred - self.y) * self.x))

            if i % 2000 == 0:
                print(f"loss at {i}th iteration : {loss}")

    def predict(self, x):
        wtx = np.dot(x, self.w)
        exp = np.exp(-wtx)
        y_prob = 1 / (1 + exp)
        predicted = np.array([1 if i > 0.5 else 0 for i in y_prob])
        return predicted


if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y.reshape(len(X), -1)
    X1 = np.ones(y.shape)
    X = np.concatenate((X1, X), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221)
    model = LogisticRegression(iterations=10000, learning_rate=0.1, x=X_train, y=y_train)
    print(f"Shape of weight matrix : {model.w.shape}")
    print(f"Shape of feature matrix : {model.x.shape}")
    print(f"Shape of output matrix : {model.y.shape}")
    model.gradient_descent()
    y_predicted = model.predict(X_test).reshape(len(X_test), -1)
    print(y_test.shape)
    accuracy = (np.sum(y_test == y_predicted) / len(y_test)) * 100
    print(f"Final weights: {model.w}")
    print(f"Accuracy of the model: {accuracy}")
