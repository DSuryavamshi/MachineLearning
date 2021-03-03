import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


class GradientDescent(object):
    def __init__(self, learning_rate, iterations, X, y):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = X.shape[0]
        self.n = X.shape[1] - 1
        self.X = X
        self.y = y.reshape(self.m, -1)
        self.w = np.zeros(((self.n + 1), 1))

    def y_hat(self):
        return np.dot(self.X, self.w)

    def loss(self, y_pred):
        cost = 1 / self.m * np.sum(np.power((y_pred - self.y), 2))
        return cost

    def gradient_descent(self):
        for epoch in range(1, self.iterations + 1):
            yhat = self.y_hat()
            loss = self.loss(y_pred=yhat)
            self.w = self.w - (self.learning_rate * (1 / self.m * np.sum((yhat - self.y) * self.X)))
            if epoch % 5000 == 0:
                print(f"Loss at epoch: {epoch}  = {loss}")
        return yhat, self.w

    def predict(self, x):
        return np.dot(x, self.w)


class NormalEquation(object):
    def __init__(self, X, y):
        self.m = X.shape[0]
        self.X = X
        self.y = y.reshape(self.m, -1)
        self.c = 0
        self.m = 0

    def normal_equation(self):
        Xtran = np.dot(self.X.T, self.X)
        xinv = np.linalg.inv(Xtran)
        final = np.dot(xinv, np.dot(self.X.T, self.y))
        self.c = final[0]
        self.m = final[1]
        ypred = self.m * self.X[:, 1:] + self.c
        return self.m, self.c

    def predict(self, x):
        return self.m * x[:, 1:] + self.c


if __name__ == '__main__':
    CWD = pathlib.Path(__file__).parent.absolute()
    dataset = pd.read_csv(f'{CWD}/Data/Salary_Data.csv')
    dataset.insert(loc=0, column='X1', value=1)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    grad_model = GradientDescent(learning_rate=0.01, iterations=200000, X=X_train, y=y_train)
    y_hat, weights = grad_model.gradient_descent()
    y_predicted_grad = grad_model.predict(X_test)
    print(f"Accuracy for Gradient Descent: {r2_score(y_true=y_test, y_pred=y_predicted_grad) * 100:.3f} %")  # 95.692 %
    print(f"GD weights: {weights}")

    norm_model = NormalEquation(X=X_train, y=y_train)
    intercept, coeff = norm_model.normal_equation()
    print(f"Normal Equation Coeff: {coeff}")
    print(f"Normal Equation Intercept: {intercept}")
    y_predicted_norm = norm_model.predict(X_test)
    print(f"Accuracy for Normal Equation: {r2_score(y_true=y_test, y_pred=y_predicted_norm) * 100:.3f} %")  # 98.817 %

    # Plotting the data
    plt.scatter(X_test[:, 1:], y_test, color='red')
    plt.plot(X_test[:, 1:], y_predicted_grad)
    plt.title("LinearRegression - Grad (No Reg)")
    plt.show()
