import pathlib

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    CWD = pathlib.Path(__file__).parent.absolute()
    dataset = pd.read_csv(f'{CWD}/Data/Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    accuracy = r2_score(y_true=y_test, y_pred=y_predicted)
    print(f"Coef: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"accuracy: {accuracy * 100:.3f} %")
