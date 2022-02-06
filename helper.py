import numpy as np


def sigmoid(x) :

    return 1.0 / (1+np.exp(-x))


def sigmoid_derv(x) :

    return  x * (1.0 - x)
