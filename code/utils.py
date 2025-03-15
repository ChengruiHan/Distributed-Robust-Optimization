import numpy as np


def projection(y):

    return np.maximum(0, y)


def projection2Omega(y,Omega):

    'Custom Settings'

    return np.clip(y, Omega[0], Omega[1])
