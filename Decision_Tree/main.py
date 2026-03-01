# CART 기반 의사결정 트리
import numpy as np
import pandas as pd
from pandas import DataFrame as df



src = "../datasets/titanic/"
train = pd.to_csv(src + "train.csv")
# 우선순위가 있는 범주형의 경우에는 매핑을 만들어서 처리
# priority_map = {"Pclass": 
#     {
#     1 : 0,
#     2: 1,
#     3: 2
# }}






# [[확률변수], [확률변수 개수], [확률]]
def calculate_probability(data):
    RV, RV_count = np.unique(data, return_counts=True)
    data_count = np.sum(RV_count)

    result = np.empty((3, RV.shape[0]))
    result[0] = RV
    result[1] = RV_count

    result[2] = RV_count / data_count

    return result


def calculate_gini(c):
    p = calculate_probability(c)
    gini = 1 - np.sum(p[2] ** 2)

    return gini


def calculate_split_gini(left_data, right_data):
    n_left = len(left_data)
    n_right = len(right_data)
    n_total = n_left + n_right

    gini_left = calculate_gini(left_data)
    gini_right = calculate_gini(right_data)

    split_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    return split_gini

# data에서 feature 하나 씩 split_gini계산해서 최고의 값을 찾음 
def find_best_split(data, features, category_feature_map:dict=None):
    # if np.issubdtype(data[:, feature_index].dtype, np.number):
    #     # 1. 연속형 처리 로직 (정렬 후 중간값 탐색)
    #     pass
    # else:
    #     # 2. 범주형 처리 로직 (부분집합 조합 탐색)
    #     pass
    # pass
    

            
            
    # 범주형 -> 숫자로 매핑해서 연속형 데이터 처럼 처리하기
    best_feature_idx = -1
    for feature, i in enumerate(features):
        
        if np.issubdtype(data[:, i], np.number):
            continue
        
        # 범주형 - 우선순위 o
        elif feature in category_feature_map:
            category = np.unique(data[:, i])
        else:
        # 데이터에 따른 처리
            # split_gini 계산
            # 최댓값인지 판단해서 best_feature_idx 업데이트
    
    # node 생성해서 값을 채움
    node = Node(left=None)
    return node


class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.threshold = None
        self.feature = None
        self.value = None


class Decsion_Tree:
    def __init__(self):
        self.root = None

    def _build_tree(self, X, y):
        # 인덱스에 특성 매핑
        feature = dict.fromkeys(range(len(X.columns)), [])
        for key, value in zip(feature.keys(), X.columns):
            feature[key] = value

        # 특성에 인덱스 매핑
        idx = dict.fromkeys(X.columns, [])
        for key, value in zip(idx.keys(), range(len(X.columns))):
            idx[key] = value

        # 1. 분할을 통해서 잎이든 노드를 root에 저장
        root = 

        # #. 최상위 노드를 반환
        return root

    def fit(self):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self):
        pass

    def predict(self):
        pass


test = np.array([[0, 1, 1, 1, 1, 1]])
# print(calculate_gini(test))


df_test = df(test, columns=["a", "b", "c", "d", "e", "f"])
print(df_test)
feature = dict.fromkeys(range(len(df_test.columns)), [])
for key, value in zip(feature.keys(), df_test.columns):
    feature[key] = value
print(feature)
