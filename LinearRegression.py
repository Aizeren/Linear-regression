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
