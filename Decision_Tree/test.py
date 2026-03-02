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
X_final = data_preprocessing(X).drop(labels="Survived", axis=1)
Y = X["Survived"]
x_np = np.array(X_final["Age"])
y_np = np.array(Y)
idx_sorted = np.argsort(x_np)
x_sorted = x_np[idx_sorted]
x_set = np.unique(np.sort(x_sorted))
y_sorted = y_np[idx_sorted]


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


# find_best_split - 연속 속성에서 최고의 분할을 찾기
def find_best_split(X, y, feature_list=None):
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
        #     feature_list[feature_idx] if feature_list else X.columns[feature_idx]
        # )  # else DataFrame으로 들어올 때, if는 그 외의 경우
        feature = feature_list[feature_idx]
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


print(find_best_split(X_final, Y, X_final.columns))


class Decsion_Tree:
    def __init__(self):
        self.root = None

    # _build_tree - 속성 중에서 분할 할 속성을 찾고 분할 후 서브 트리를 통해서 재귀적으로 트리 구축
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
