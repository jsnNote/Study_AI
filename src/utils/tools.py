import numpy as np
import pandas as pd
from pandas import DataFrame
from .metrics import *

def mode(data):
    if len(data) == 0:
        return None
    
    values, counts = np.unique(data, return_counts=True)
    max_idx = np.argmax(counts)

    return values[max_idx]

# find_best_split - 연속 속성에서 최고의 분할을 찾 -> 하나의 속성에 대해서도 동작하게 수정
def find_best_split(X, y, features=None):
    # 모든 입력을 연속 속성으로 하고 있음.
    # 정렬 후 두 데이터의 중간값을 기준으로 지니계수를 구함.
    best_split_gini = np.inf
    best_threshold = -1
    best_feature = None
    split_info = {"feature": [], "threshold": []}
    X_np = np.array(X)
    y_np = np.array(y)
    
    for feature_idx in range(X.shape[1]):
        column_values = X_np[:, feature_idx]
        idx_sorted = np.argsort(column_values)  # x가 DataFrame이면 안됨
        x_sorted = column_values[idx_sorted]
        y_sorted = y_np[idx_sorted]
        v = np.unique(np.sort(x_sorted))
        thresholds = (v[:-1] + v[1:]) / 2
        # feature = (
        #     features[feature_idx] if features else X_columns[feature_idx]
        # )  # else DataFrame으로 들어올 때, if는 그 외의 경우
        feature = features[feature_idx]
          # else DataFrame으로 들어올 때, if는 그 외의 경우
        for t in thresholds:
            left_data = y_sorted[np.where(x_sorted <= t)]
            right_data = y_sorted[np.where(x_sorted > t)]
            split_gini = calculate_split_gini(left_data, right_data)
            if best_split_gini > split_gini:
                best_split_gini = split_gini
                best_threshold = t
                best_feature = feature
    split_info["feature"] = best_feature
    split_info["threshold"] = best_threshold

    return split_info


