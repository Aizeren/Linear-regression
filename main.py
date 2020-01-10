from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from Matrix import Matrix
from LinearRegression import LinearRegression
from functions import train_test_split
from functions import rmse

if __name__ == "__main__":
    X, y = make_regression(n_samples=3000, n_features=1, noise = 10)
    X, y = X.tolist(), y.tolist()

    X = Matrix(X)
    y = Matrix(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.7)
    lnrReg = LinearRegression()
    lnrReg.fit(X_train, y_train)
    prediction = lnrReg.predict(X_test)

    print('RMSE = ', rmse(y_test, prediction))

    plt.scatter([row[0] for row in X_test.arr], y_test.arr)
    plt.scatter([row[0] for row in X_test.arr], prediction.arr)
    plt.xlabel('X') 
    plt.ylabel('y')

    plt.show() 
