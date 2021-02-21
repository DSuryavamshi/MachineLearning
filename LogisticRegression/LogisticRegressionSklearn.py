from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y.reshape(len(X), -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test).reshape(len(X_test), -1)
    accuracy = (np.sum(y_test == prediction) / len(y_test)) * 100
    print(f"Accuracy of the model: {accuracy}")
