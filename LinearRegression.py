from Matrix import Matrix
from copy import deepcopy

class LinearRegression():
    def __init__(self):
        self.coef=Matrix()

    def fit(self, X: Matrix, y: Matrix):
        X = X.add_ones_left()
        self.coef = (((X.transp() * X).inverse())*X.transp())*y

    def predict(self, X):
        X = X.add_ones_left()
        return X*self.coef

    def L2(self, X: Matrix, y: Matrix, lmbd: float):
        X = X.add_ones_left()
        E = Matrix(nrows=X.ncolumns, ncolumns=X.ncolumns)
        for i in range(0, E.nrows):
            E.arr[i][i] = lmbd
        self.coef = ((((X.transp() * X) + E).inverse())*X.transp())*y
