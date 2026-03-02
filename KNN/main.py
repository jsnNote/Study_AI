import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

from sklearn.preprocessing import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

src = "../datasets/titanic/"

# 1. 데이터 전처리
train = pd.read_csv(src + "train.csv")


def DataPreprocessing(data: DataFrame):
    _data = data.copy()
    _data["Sex_numeric"] = _data["Sex"].map({"male": 0, "female": 1})
    _data["Title"] = _data["Name"].str.extract(" ([A-Za-z]+)\.")
    median_ages = _data.groupby("Title")["Age"].transform("median")
    _data["Age"] = _data["Age"].fillna(median_ages)
    _data["FamilySize"] = _data["SibSp"] + _data["Parch"] + 1
    # Ms 데이터가 하나 있는데 값이 없어서 제거
    _data = _data.select_dtypes(exclude=["str"])
    _data = _data.dropna(axis=0)
    _data = _data.dropna(axis=1)
    return _data


X = train.transform(DataPreprocessing)
X = X.drop(labels="Survived", axis=1)
Y_train = train["Survived"].values

# 특성 선택
used_SelectKBest = False
feat_sel = None

if used_SelectKBest:
    #   - SelectKBest ✅
    feat_sel = SelectKBest(f_classif, k=3)
    X_selected = feat_sel.fit_transform(X)
else:
    #   - SelectKBest ❌
    feat_sel = ["Pclass", "Sex_numeric", "Age"]
    # feat_sel = ["Pclass", "Age"]
    X_selected = X[feat_sel]

minmax_sclaer = MinMaxScaler()  # 특성의 범위가 불균형해서 0~1 범위로 스케일링
minmax_sclaer.fit(X_selected)
X_scaled = minmax_sclaer.transform(X_selected)

x_train, x_valid, y_train, y_valid = train_test_split(
    X_scaled, Y_train, test_size=0.2, random_state=42
)


# 2. 모델 구성
def manhattan(x1, x2):
    return np.sum(np.abs(x1 - x2), axis=1)


def minkowski(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p, axis=1) ** (1 / p)


def euclidean(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))


class KNN:
    def __init__(self, k: int = 3, metric=minkowski, p: int = 2):
        self.k = k
        self.metric = None
        self.p = p
        self.X = None
        allowed_metrics = [manhattan, minkowski, euclidean]
        if metric in allowed_metrics:
            self.metric = metric
        else:
            raise ValueError(
                "metric은 manhattan, minkowski, euclidean 중 하나여야 합니다."
            )

    def _get_distance(self, x1, x2):
        if self.metric == minkowski:
            return self.metric(x1, x2, self.p)
        return self.metric(x1, x2)

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, data: np.ndarray):
        if data.ndim == 1:
            data = data.reshape((1, -1))

        size = data.shape[0]
        predict_value = np.empty(size, dtype=int)

        for item, idx in zip(data, range(size)):
            distance = self._get_distance(self.X_train, item)
            distance_list = [(d, l) for d, l in zip(distance, self.Y_train)]
            distance_list.sort(key=lambda x: x[0])
            k_label_list = distance_list[: self.k]
            label_cnt = dict.fromkeys(set([v[1] for v in k_label_list]), 0)

            for value in k_label_list:
                label_cnt[value[1]] += 1

            for k, v in label_cnt.items():
                if max(label_cnt.values()) == v:
                    predict_value[idx] = k

        return predict_value


# 3. 학습

# k 하이퍼 파라미터 선택(튜닝)
max_score = 0
best_k = 0
for i in range(1, 31):
    model = KNN(k=i, metric=euclidean)
    model.fit(x_train, y_train)

    score = np.mean(y_valid == model.predict(x_valid))

    if max_score < score:
        max_score = score
        best_k = i

model = KNN(k=best_k, metric=euclidean)
model.fit(x_train, y_train)

# test = pd.read_csv(src + "test.csv")
# X_test = train.transform(DataPreprocessing)
# if used_SelectKBest:
#     #   - SelectKBest ✅
#     X_test_selected = feat_sel.fit_transform(X_test)
# else:
#     #   - SelectKBest ❌
#     X_test_selected = X_test[feat_sel]

# minmax_sclaer = MinMaxScaler()  # 특성의 범위가 불균형해서 0~1 범위로 스케일링
# minmax_sclaer.fit(X_test_selected)
# X_test_scaled = minmax_sclaer.transform(X_test_selected)

# 4. 검증
print(np.mean(y_valid == model.predict(x_valid)))


# ⭐ KNeighborsClassifier 사용
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(x_train, y_train)

# 4. 예측
print(np.mean(y_valid == knn.predict(x_valid)))

# survived_idx = np.where(model.Y_train == 1)
# survived_data = model.X_train[survived_idx]
# dead_idx = np.where(model.Y_train == 0)
# dead_data = model.X_train[dead_idx]

# label = {1: "survived", 0: "dead"}
# plt.scatter(survived_data[:, 0], survived_data[:, 1], c="green")
# plt.scatter(dead_data[:, 0], dead_data[:, 1], c="red")
# plt.scatter(x_valid[1, 0], x_valid[1, 1], c="blue", s=100)
# print(model.predict(x_valid[1]))
# plt.show()
