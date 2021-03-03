import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RegularizedLinearRegressionGrad:
    def __init__(self, learning_rate, lmbda, iterations, X, y):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lmbda = lmbda
        self.m = X.shape[0]
        self.n = X.shape[1] - 1
        self.X = X
        self.y = y.reshape(self.m, -1)
        self.w = np.zeros(((self.n + 1), 1))

    def y_hat(self):
        return np.dot(self.X, self.w)

    def loss(self, y_pred):
        loss = (1 / self.m) * (np.sum(y_pred - self.y) + self.lmbda * (np.sum(self.w[1:, :])))
        return loss

    def gradient_descent(self):
        for epoch in range(1, self.iterations + 1):
            yhat = self.y_hat()
            loss = self.loss(y_pred=yhat)
            self.w[0] = self.w[0] - (self.learning_rate * (1 / self.m * np.sum((yhat - self.y) * self.X)))
            self.w[1:] = self.w[1:] - (self.learning_rate * (
                    1 / self.m * np.sum((yhat - self.y) * self.X) + (self.lmbda / self.m) * self.w[1:]))
            if epoch % 5000 == 0:
                print(f"Loss at epoch: {epoch}  = {loss}")
        return yhat, self.w

    def predict(self, x):
        return np.dot(x, self.w)


class RegularizedLinearRegressionNorm:
    def __init__(self, X, y, lmbda):
        self.m = X.shape[0]
        self.X = X
        self.n = X.shape[1] - 1
        self.y = y.reshape(self.m, -1)
        self.c = 0
        self.m = 0
        self.lmbda = lmbda

    def normal_equation(self):
        ident = np.eye(self.n + 1)
        ident[0, 0] = 0
        lmbda_ident = self.lmbda * ident
        Xtran = np.dot(self.X.T, self.X) + lmbda_ident
        xinv = np.linalg.inv(Xtran)
        final = np.dot(xinv, np.dot(self.X.T, self.y))
        self.c = final[0]
        self.m = final[1]
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
    grad_model = RegularizedLinearRegressionGrad(learning_rate=0.01, lmbda=0.01, iterations=200000, X=X_train,
                                                 y=y_train)
    y_hat, weights = grad_model.gradient_descent()
    y_predicted_grad = grad_model.predict(X_test)
    print(f"Accuracy for Gradient Descent: {r2_score(y_true=y_test, y_pred=y_predicted_grad) * 100:.3f} %")  # 96.642 %
    print(f"GD weights: {weights}")

    norm_model = RegularizedLinearRegressionNorm(X=X_train, y=y_train, lmbda=-0.8)  # 98.858 %
    intercept, coeff = norm_model.normal_equation()
    print(f"Normal Equation Coeff: {coeff}")
    print(f"Normal Equation Intercept: {intercept}")
    y_predicted_norm = norm_model.predict(X_test)
    print(f"Accuracy for Normal Equation: {r2_score(y_true=y_test, y_pred=y_predicted_norm) * 100:.3f} %")  # 96.783 %

    # Plotting the data
    plt.scatter(X_test[:, 1:], y_test, color='red')
    plt.plot(X_test[:, 1:], y_predicted_grad)
    plt.title("LinearRegression - Grad (With Reg)")
    plt.show()