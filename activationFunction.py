import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

# https://stackoverflow.com/questions/38597587/how-to-perform-an-operation-on-every-element-in-a-numpy-matrix