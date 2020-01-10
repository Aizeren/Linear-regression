from Matrix import Matrix
from copy import deepcopy
from math import sqrt

def train_test_split(X: Matrix, y: Matrix, train_size):
    train_size = round(X.nrows * train_size)
    test_size = X.nrows - train_size

    X_train = deepcopy(X)
    X_test = deepcopy(X)

    y_train = deepcopy(y)
    y_test = deepcopy(y)

    X_train.nrows = train_size
    X_test.nrows = test_size

    y_train.nrows = train_size
    y_test.nrows = test_size

    X_train.arr = X.arr[:train_size]
    X_test.arr = X.arr[train_size:]

    y_train.arr = y.arr[:train_size]
    y_test.arr = y.arr[train_size:]
    
    return X_train, X_test, y_train, y_test

def rmse(y: Matrix, pred: Matrix):
    if y.nrows != pred.nrows:
        raise ValueError('Number of rows in y not equals to number of rows in prediction')
    res=0
    for i in range(0, y.nrows):
        res += (y.arr[i][0] - pred.arr[i][0])**2
    res /= y.nrows
    res = sqrt(res)

    return res
