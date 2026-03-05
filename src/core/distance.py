import numpy as np
import pandas as pd
from pandas import DataFrame

def manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)


def minkowski(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)


def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))