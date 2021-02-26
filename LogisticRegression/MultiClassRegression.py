import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from LogisticRegression.LogisticRegressionScratch import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class MultiClassRegression(LogisticRegression):
    def __init__(self, learning_rate, iterations, x, y):
        super().__init__(learning_rate, iterations, x, y)
        self.classes = np.unique(y)

    def multiclass_algo(self):
        main_output = self.y
        ohe = OneHotEncoder()
        test = ohe.fit_transform(main_output)
        print(test)
        # for cls in self.classes:
        #     new_y = self.y
        #     new_y[new_y != cls] = -999
        #     if new_y.__contains__(0):
        #         print(new_y)
        #     print(new_y.shape)
        #     pass

    def gradient_descent(self):
        for i in range(1, self.iter):
            ypred = self.y_hat()
            final_y = ypred
            loss = self.loss(ypred=ypred)
            self.w = self.w - ((self.lr / self.m) * np.sum((ypred - self.y) * self.x))

            if i % 2000 == 0:
                print(f"loss at {i}th iteration : {loss}")


if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=3)
    y = y.reshape(len(X), -1)
    X1 = np.ones(y.shape)
    X = np.concatenate((X1, X), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221)
    model = MultiClassRegression(iterations=10000, learning_rate=0.1, x=X_train, y=y_train)
    print(f"Shape of weight matrix : {model.w.shape}")
    print(f"Shape of feature matrix : {model.x.shape}")
    print(f"Shape of output matrix : {model.y.shape}")
    model.multiclass_algo()
