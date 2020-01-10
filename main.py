from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from Matrix import Matrix
from LinearRegression import LinearRegression
from functions import train_test_split
from functions import rmse

if __name__ == "__main__":
    X, y = make_regression(n_samples=20, n_features=3, noise = 10)
    X, y = X.tolist(), y.tolist()

    X = Matrix(X)
    y = Matrix(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.7)
    lnrReg = LinearRegression()
    lnrReg.fit(X_train, y_train)
    prediction = lnrReg.predict(X_test)

    print('RMSE = ', rmse(y_test, prediction))

    beg = 0
    end = 100

    func = lambda i: i if i>0 else (-1/i if i<0 else 1)
    lmbd = [func(i) for i in range(beg, end, 1)]
    coefs = [[]]*len(lnrReg.coef.arr)
    for col in range(0, len(lnrReg.coef.arr)):
        coefs[col] = [0]*len(lmbd)

    for i in range(beg, end, 1):
        
        lnrReg.L2(X_train, y_train, i)
        for j in range(0, len(coefs)):
            coefs[j][i] = lnrReg.coef.arr[j][0]

    # plt.scatter([row[0] for row in X_test.arr], y_test.arr)
    # plt.plot([row[0] for row in X_test.arr], prediction.arr, 'r')

    # plt.xlabel('X') 
    # plt.ylabel('y')
    for i in range(0, len(coefs)):
        plt.plot(lmbd, coefs[i])
    plt.xlabel('lambda') 
    plt.ylabel('coefficients')

    plt.show() 
