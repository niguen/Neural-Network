from math import exp

import numpy as np


class matrix:

    def __init__(self, rows, columns, values=None):

        self.rows = rows
        self.columns = columns
        if values is None:
            self.values = []
            for i in range(rows):
                row = []
                for j in range(columns):
                    row.append(0)
                self.values.append(row)
        else:
            self.values = []
            for i in range(rows):
                row = []
                for j in range(columns):
                    row.append(values[j + (i * columns)])
                self.values.append(row)
        pass

    def toArray(self):
        result = []
        for i in range(self.rows):
            for j in range(self.columns):
                result.append(self.values[i][j])
        return result


    def __str__(self):
        string = ""
        for i in range(self.rows):
            string += str(self.values[i]) + "\n"
        return string

    def random(self, incommingConnections):
        for i in range(self.rows):
            for j in range(self.columns):
                self.values[i][j] = np.random.normal(0.0, pow(incommingConnections, -0.5))
        return self

    def T(self):
        col = self.columns
        self.columns = self.rows
        self.rows = col

        newValues = []
        for i in range(self.rows):
            row = []
            for j in range(self.columns):
                row.append(self.values[j][i])
            newValues.append(row)
        self.values = newValues
        return self

    def __neg__(self):
        for i in range(self.rows):
            for j in range(self.columns):
                self.values[i][j] = -1 * self.values[i][j]

        return self

    def __add__(self, other):
        if type(other) is float:
            for i in range(self.rows):
                for j in range(self.columns):
                    self.values[i][j] = other + self.values[i][j]

            return self
        for i in range(self.rows):
            for j in range(self.columns):
                self.values[i][j] = other.values[i][j] + self.values[i][j]

        return self

    def __sub__(self, other):
        if other.columns != self.columns or other.rows != self.rows:
            raise ValueError("Multiplication not possible! Matrix dimensions do not match.")

        result = matrix(self.rows, self.columns)
        for i in range(self.rows):
            for j in range(self.columns):
                result.values[i][j] = self.values[i][j] - other.values[i][j]
        return result

    def __mul__(self, other):
        if (type(other) is int) or (type(other) is float):
            for i in range(self.rows):
                for j in range(self.columns):
                    self.values[i][j] *= other
        else:
            for i in range(self.rows):
                for j in range(self.columns):
                    self.values[i][j] *= other.values[i][j]
        return self

    def __truediv__(self, other):
        for i in range(self.rows):
            for j in range(self.columns):
                self.values[i][j] /= other
        return self

    def set(self, row, column, value):
        self.values[row][column] = value
        return self

    def __IADD__(self, other):
        for i in range(self.rows):
            for j in range(self.columns):
                self.values[i][j] = self.values[i][j] + other.values[i][j]
        return self

    @staticmethod
    def dot(matrix1, matrix2):
        if matrix1.columns != matrix2.rows:
            print("Matrix1: Rows: " + str(matrix1.rows) + ", Columns: " + str(matrix1.columns))
            print("Matrix2: Rows: " + str(matrix2.rows) + ", Columns: " + str(matrix2.columns))
            raise ValueError("Multiplication not possible! Matrix dimensions do not match.")

        result = matrix(matrix1.rows, matrix2.columns)
        for i in range(matrix1.rows):
            for j in range(matrix2.columns):

                for x in range(matrix1.columns):
                    result.values[i][j] += matrix1.values[i][x] * matrix2.values[x][j]
        return result

    @staticmethod
    def transpose(matrix1):
        result = matrix(matrix1.columns, matrix1.rows)

        newValues = []
        for i in range(matrix1.columns):
            row = []
            for j in range(matrix1.rows):
                row.append(matrix1.values[j][i])
            newValues.append(row)
        result.values = newValues
        return result
        pass

    @staticmethod
    def exp(matrix1):
        for i in range(matrix1.rows):
            for j in range(matrix1.columns):
                matrix1.values[i][j] = exp(matrix1.values[i][j])
        return matrix1

    @staticmethod
    def sigmoid(matrix1):
        for i in range(matrix1.rows):
            for j in range(matrix1.columns):
                matrix1.values[i][j] = 1 / (1 + exp(-matrix1.values[i][j]))

        return matrix1

    @staticmethod
    def asfarray(array):
        result = matrix(1, len(array))
        for i in range(len(array)):
            result.values[0][i] = float(array[i])

        return result
