from cmath import e

def sigmoid(x):
    return 1 / (1 + e ** (-x))

# https://stackoverflow.com/questions/38597587/how-to-perform-an-operation-on-every-element-in-a-numpy-matrix