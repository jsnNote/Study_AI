import numpy as np
import pandas as pd
from pandas import DataFrame


def accuracy_score(predict, solution):
    return np.mean(predict == solution)


# [[확률변수], [확률변수 개수], [확률]]
def _calculate_probability(data):
    RV, RV_count = np.unique(data, return_counts=True)
    data_count = np.sum(RV_count)

    result = {"random_variable": [], "random_variable_count": [], "probabiliry": []}
    result["random_variable"] = RV
    result["random_variable_count"] = RV_count
    result["probability"] = RV_count / data_count

    return result


def calculate_gini(c):
    p = _calculate_probability(c)
    gini = 1 - np.sum(p["probability"] ** 2)
    return gini


def calculate_split_gini(left_data, right_data):
    n_left = len(left_data)
    n_right = len(right_data)
    n_total = n_left + n_right
    gini_left = calculate_gini(left_data)
    gini_right = calculate_gini(right_data)

    split_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    return split_gini