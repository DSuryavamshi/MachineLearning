import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class MultiClassRegression():
    def __init__(self, learning_rate, iterations, x, y):
        self.x = x
        self.y = y
        self.m, self.n = x.shape
        self.lr = learning_rate
        self.iter = iterations
        self.classes = np.unique(y)
        self.classes.sort()
        self.w = []
        for i in range(self.classes.shape[0]):
            self.w.append(np.zeros((self.n, 1)))
        ohe = OneHotEncoder()
        self.new_y = ohe.fit_transform(self.y).toarray()
        self.ys = []
        for i in range(self.new_y.shape[1]):
            self.ys.append(np.array(self.new_y[:, i]))
            self.ys[i] = self.ys[i].reshape(self.m, -1)

    def gradient_descent(self):
        for i in range(1, self.iter):
            ypred = self.y_hat()
            loss = self.loss(ypred=ypred)
            if i % 2000 == 0:
                print(f"loss at {i}th iteration : {loss}")
            for j in range(len(self.w)):
                self.w[j] = self.w[j] - ((self.lr / self.m) * np.sum((ypred[j] - self.ys[j]) * self.x))

    def y_hat(self):
        y_prob = []
        for weight in self.w:
            wtx = np.dot(self.x, weight)
            exp = np.exp(-wtx)
            y_i = 1 / (1 + exp)
            y_prob.append(y_i)
        return y_prob

    def loss(self, ypred):
        overall_loss = []
        for i in range(len(ypred)):
            checkforzero = ypred[i]  # To handle log10(0) issue
            checkforzero[checkforzero == 0] = 10 ** -10
            hx = np.where(checkforzero > 10e-10, np.log10(checkforzero), -10)
            hx[hx == -10] = 0

            checkforzero = 1 - ypred[i]  # To handle log10(0) issue
            checkforzero[checkforzero == 0] = 10 ** -10
            h_1x = np.where(checkforzero > 10e-10, np.log10(checkforzero), -10)
            loss = -(1 / self.m) * np.sum((self.ys[i] * hx) + ((1 - self.ys[i]) * h_1x))
            overall_loss.append(loss)
        return np.average(overall_loss)

    def predict(self, x):
        final_prediction = []
        for weight in self.w:
            wtx = np.dot(x, weight)
            exp = np.exp(-wtx)
            y_i = 1 / (1 + exp)
            predicted = y_i
            final_prediction.append(predicted)
        final_pred_arr = np.concatenate(final_prediction, axis=1)
        print(final_prediction[1].shape)
        final_pred = np.array([self.classes[np.argmax(final_pred_arr[i])] for i in range(final_pred_arr.shape[0])])
        return final_pred


if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=3)
    y = y.reshape(len(X), -1)
    X1 = np.ones(y.shape)
    X = np.concatenate((X1, X), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221)
    model = MultiClassRegression(iterations=20000, learning_rate=0.1, x=X_train, y=y_train)
    print(f"Shape of weight matrix : {model.w[0].shape}")
    print(f"Shape of feature matrix : {model.x.shape}")
    print(f"Shape of output matrix : {model.y.shape}")
    model.gradient_descent()
    y_predicted = model.predict(x=X_test)
    y_comp = pd.DataFrame({
        'y_pred': list(y_predicted),
        'y_test': list(y_test)
    })
    y_comp['Result'] = y_comp['y_pred'] == y_comp['y_test']
    y_comp.to_csv('comp.csv', index=False)
    accuracy = (np.sum(y_comp['Result'] == True) / y_test.shape[0]) * 100
    print(f"Accuracy of the model: {accuracy}")
