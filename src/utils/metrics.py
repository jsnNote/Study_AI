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

# model은 학습을 한 상태
class BinaryClassifierPerformance:
    def __init__(self, prediction, actual):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.__calculate_confusion_matrix(prediction, actual)
        
    def __calculate_confusion_matrix(self, prediction, actual):
        actual_np = np.array(actual) # series를 타입변환
        # 예측 값이 postive(1)인 경우
        p_mask = (prediction == 1)
        # 같을 때 1, 다른 때 0
        self.TP = np.sum((prediction[p_mask] == actual_np[p_mask]))
        self.FP = np.sum(p_mask) - self.TP
        
        # 예측 값이 negative(0)인 경우
        # 같을 때 1, 다른 때 0
        n_mask = (prediction == 0)
        self.TN = np.sum(prediction[n_mask] == actual_np[n_mask])
        self.FN = np.sum(n_mask) - self.TN

    @property
    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)
    
    @property
    def precision(self):
        return (self.TP) / (self.TP + self.FP)
    
    @property
    def recall(self):
        return (self.TP) / (self.TP + self.FN)
    
    @property
    def F1_score(self):
        return (2 * self.precision * self.recall) / (self.precision + self.recall)