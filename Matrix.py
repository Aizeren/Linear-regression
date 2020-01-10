from copy import deepcopy

class Matrix():
    def __init__(self, arr: list = None, nrows: int = None, ncolumns: int = None):
        if arr is None:
            if nrows is not None and ncolumns is not None:
                self.nrows=nrows
                self.ncolumns=ncolumns
                self.arr = [[]]*nrows
                for col in range(0, nrows):
                    self.arr[col] = [0]*ncolumns
            else:
                self.arr=None
                self.nrows=None
                self.ncolumns=None
        else:
            self.nrows = len(arr)
            if not isinstance(arr[0], list):
                self.ncolumns = 1
                self.arr = [[]]*self.nrows
                for row in range(0, self.nrows):
                    self.arr[row] = [arr[row]]
            else:
                self.ncolumns = len(arr[0])
                self.arr=arr
            

    def transp(self):
        transp_arr = Matrix(nrows = self.ncolumns, ncolumns = self.nrows)
        arr=transp_arr.arr
        for row in range(0, transp_arr.nrows):
            for col in range(0, transp_arr.ncolumns):
                arr[row][col] = self.arr[col][row]

        return transp_arr
        

    def inverse(self):
        matrix = deepcopy(self)
        det = matrix.find_det()
        if det == 0:
            raise ValueError('Matrix is not invertable')
        arr = matrix.arr

        # special case for 2x2 matrix:
        if matrix.ncolumns == 2:
            return Matrix([[arr[1][1] / det, -1 * arr[0][1] / det],
                    [-1 * arr[1][0] / det, arr[0][0] / det]])

        res = []
        for r in range(matrix.ncolumns):
            res_row = []
            for c in range(matrix.ncolumns):
                minor = matrix.make_minor(r, c)
                res_row.append(((-1) ** (r + c)) * minor.find_det())
            res.append(res_row)

        res = Matrix(res)
        res = res.transp()
        res = mul_const_with_matrix(res, 1 / det)

        return res

    def make_minor(self, i, j):
        minor = deepcopy(self)
        minor.ncolumns = minor.ncolumns - 1
        minor.nrows = minor.nrows - 1
        arr = minor.arr
        minor.arr = [row[:j] + row[j + 1:] for row in (arr[:i] + arr[i + 1:])]

        return minor

    def find_det(self):
        if self.nrows != self.ncolumns:
            raise ValueError('Matrix is not square')
        det = 1
        triang_arr = self.triangular()
        arr = triang_arr.arr

        for i in range(0, self.nrows):
            det = det * arr[i][i]
        
        return det

    def add_ones_left(self):
        res = Matrix(nrows=self.nrows, ncolumns=self.ncolumns+1)
        for row in range(0, res.nrows):
            res.arr[row][0] = 1
            for col in range(1, res.ncolumns):
                res.arr[row][col] = self.arr[row][col-1]
        
        return res

    def __mul__(self, other):
        if self.ncolumns != other.nrows:
            raise ValueError('Size of matrices is wrong')
        res = Matrix(nrows = self.nrows, ncolumns = other.ncolumns)

        for row in range(0, self.nrows):
            for col in range(0, other.ncolumns):
                for i in range(0, self.ncolumns):
                    res.arr[row][col] += self.arr[row][i] * other.arr[i][col]
        
        return res

    def __add__(self, other):
        if self.nrows != other.nrows:
            raise ValueError('self.nrows != other.nrows')
        if self.ncolumns != other.ncolumns:
            raise ValueError('self.ncolumns != other.ncolumns')
        res = Matrix(nrows=self.nrows, ncolumns=self.ncolumns)
        for row in range(0, self.nrows):
            for col in range(0, self.ncolumns):
                res.arr[row][col] = self.arr[row][col] + other.arr[row][col]
        
        return res


    def __str__(self):
        arrstr = ''
        for row in range(0, self.nrows):
            for column in range(0, self.ncolumns):
                arrstr = arrstr + str('{:.2f}'.format(self.arr[row][column])) + '\t'
            arrstr = arrstr + '\n'
        return arrstr

    def triangular(self):
        triang_arr = deepcopy(self)
        arr = triang_arr.arr
        for k in range(self.nrows - 1):
            arr = bubble_max_row(arr, k)
            for i in range(k + 1, self.nrows):
                div = arr[i][k] / arr[k][k]
                for j in range(k, self.nrows):
                    arr[i][j] -= div * arr[k][j]

        return triang_arr
        

def bubble_max_row(arr, col):
    ret = arr
    max_element = ret[col][col]
    max_row = col
    for i in range(col + 1, len(ret)):
        if abs(ret[i][col]) > abs(max_element):
            max_element = ret[i][col]
            max_row = i
    if max_row != col:
        ret[col], ret[max_row] = ret[max_row], ret[col]

    return ret


def mul_const_with_matrix(matrix: Matrix, const):
    res = deepcopy(matrix)
    for i in range(matrix.nrows):
        for j in range(matrix.ncolumns):
            res.arr[i][j] = matrix.arr[i][j] * const
    return res