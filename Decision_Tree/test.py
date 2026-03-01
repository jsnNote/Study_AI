import numpy as np
import pandas as pd
from pandas import DataFrame

src = "../datasets/titanic/"
# 1. 데이터 전처리
train = pd.read_csv(src + "train.csv")


def data_preprocessing(df: DataFrame):
    new_df = df.copy()
    new_df["Sex_numeric"] = new_df["Sex"].map({"male": 0, "female": 1})
    new_df["Title"] = new_df["Name"].str.extract(" ([A-Za-z]+)\.")
    median_ages = new_df.groupby("Title")["Age"].transform("median")
    new_df["Age"] = new_df["Age"].fillna(median_ages)
    new_df["FamilySize"] = new_df["SibSp"] + new_df["Parch"] + 1
    # Ms 데이터가 하나 있는데 값이 없어서 제거

    cols_to_drop = [
        "PassengerId",
        "Ticket",
        "Name",
        "Cabin",
        "Sex",
        "Title",
        "Embarked",
    ]
    new_df = new_df.drop(cols_to_drop, axis=1)
    return new_df


# 데이터 형태
# 연속형
# [1, 12, 3, 4]
# 트리에서 처리

# 범주형
# - 우선순위 o
#   - 직접 순위 지정
#   {"feature1": 1, "feature2": 2, "feature3": 3}
# - 우선순위 x
#   - 원핫인코딩


def one_hot_encoding(data, feature=None):
    result = {"columns": [], "encoding": {}}
    columns = set(data)
    encoding_size = len(columns)

    if feature:
        result["columns"] = [f"{feature}_{column}" for column in columns]
    else:
        result["columns"] = columns

    result["encoding"] = dict.fromkeys(columns, [])
    for i, column in enumerate(columns):
        encoding_array = [0] * encoding_size  # 객체의 메모리 할당
        encoding_array[i] = 1  # 해당하는 값에
        result["encoding"][column] = encoding_array
    return result


# 우선순위 데이터 처리 함수
def ordinal_data_preprocessing(df: DataFrame, priority: dict, feature=None):
    rows = []
    for v in df[feature]:
        new_row = [priority[v]]
        rows.append(new_row)
    _df = DataFrame(rows, columns=[f"{feature}_numeric"], index=df.index)
    new_df = pd.concat([df, _df], axis=1)

    return new_df


# pclass_priority = {1: 1, 2: 2, 3: 3}
# print(ordinal_data_preprocessing(X, pclass_priority, feature="Pclass"))


# 범주형 데이터 처리 함수 - 데이터가 3개 이상인 경우 사용
def categorical_data_preprocessing(df: DataFrame, data: dict, feature=None):
    rows = []
    for v in df[feature]:
        new_row = data["encoding"][v]
        rows.append(new_row)

    _df = DataFrame(rows, columns=data["columns"], index=df.index)
    new_df = pd.concat([df, _df], axis=1)

    return new_df


drop_index = train["Embarked"].dropna(axis=0).index
train_final = train.iloc[drop_index]

embarked_encoding = one_hot_encoding(train_final["Embarked"], "Embarked")
X = categorical_data_preprocessing(train_final, embarked_encoding, feature="Embarked")
X_final = data_preprocessing(X)


print(X_final)


# [[확률변수], [확률변수 개수], [확률]]
def _calculate_probability(data):
    RV, RV_count = np.unique(data, return_counts=True)
    data_count = np.sum(RV_count)

    result = np.empty((3, -1))
    result[0] = RV
    result[1] = RV_count

    result[2] = RV_count / data_count

    return result


def calculate_gini(c):
    p = _calculate_probability(c)
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


def find_best_split():
    # 잎노드에단일값인지 판단해서 value에 값넣기
    # 중간 노드는 규칙을 만드는 거기 떄문에 value를 넣을 필요 없음.
    best_split_gini = -1
    split_info = {"value": [], "feature": [], "threshold": []}

    for feature_idx in range():
        pass

    node = Node(value, feature, threshold)
    return node


class Decsion_Tree:
    def __init__(self):
        self.root = None

    def _build_tree(self, X: np.ndarray, y):  # 수치형 feature로 처리
        # 중지 기준
        # 1. 대상 속성이 단일값을 가지면
        # 2. 데이터가 없을 때
        # 기술 속성이 없을 때
        return_flag = False
        self.root = find_best_split(X, y)  # 단일 데이터일 경우에는 반환
        if self.root.Threshold is None:
            return_flag = True

        if not return_flag:  # 종료 조건
            subset_X = X.drop(labels=self.root.feature, axis=1)
            self.root.left = self._build_tree(subset_X, y)
            self.root.right = self._build_tree(subset_X, y)

        else:
            return self.root

        return self.root

    def fit(self):
        pass
