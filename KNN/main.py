import sys
import os

# 현재 파일(main.py)의 부모(A)의 부모(my_ml_project) 경로를 가져옴
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 파이썬 모듈 검색 경로 최상단에 추가
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

from sklearn.preprocessing import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from src import KNN, accuracy_score, save_trained_model, load_trained_model
from src.core import minkowski, manhattan, euclidean
from src.utils.metrics import BinaryClassifierPerformance
import pickle

path = "../datasets/titanic/"
# 1. 데이터 전처리
train = pd.read_csv(path + "train.csv")


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


X = data_preprocessing(train)
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

x_train, x_validation, y_train, y_validation = train_test_split(
    X_scaled, Y_train, test_size=0.2, random_state=42
)

# k 하이퍼 파라미터 선택(튜닝)
max_score = 0
best_k = 0
for i in range(1, 31):
    model = KNN(k=i, metric=euclidean)
    model.fit(x_train, y_train)

    score = np.mean(y_validation == model.predict(x_validation))

    if max_score < score:
        max_score = score
        best_k = i

# 2. 모델 구성
model = KNN(k=best_k, metric=euclidean)
# 3. 모델 학습
model.fit(x_train, y_train)
save_trained_model(model, "./knn.pickle")

knn_per = BinaryClassifierPerformance(model.predict(x_validation), y_validation)
print("KNN")
print(f"Accuracy: {knn_per.accuracy}")
print(f"Precision: {knn_per.precision}")
print(f"Recall: {knn_per.recall}")
print(f"F1-score: {knn_per.F1_score}")

# 4. 검증
print(accuracy_score(y_validation, model.predict(x_validation)))

# 5. 예측값 저장
data = {"prediction": model.predict(x_validation)}

exit()
# ⭐ KNeighborsClassifier 사용
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(x_train, y_train)

# 4. 예측


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
