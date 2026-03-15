import sys
import os

# 현재 파일(main.py)의 부모(A)의 부모(my_ml_project) 경로를 가져옴
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 파이썬 모듈 검색 경로 최상단에 추가
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src import DecsionTree
from src import categorical_data_preprocessing, one_hot_encoding, accuracy_score

path = "../datasets/titanic/"
train = pd.read_csv(path + "train.csv")

# 1. 데이터 전처리
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

# 범주형 -> 연속형 데이터로 만듦(처리가 쉬움. 단, 과적합에 취약)
# - 우선순위 o
#   - 직접 순위 지정
#   {"feature1": 1, "feature2": 2, "feature3": 3}
# - 우선순위 x
#   - 원핫인코딩

drop_index = train["Embarked"].dropna(axis=0).index
train_final = train.iloc[drop_index]

embarked_encoding = one_hot_encoding(train_final["Embarked"], "Embarked")
X = categorical_data_preprocessing(train_final, embarked_encoding, feature="Embarked")
X_final = data_preprocessing(X).drop(labels="Survived", axis=1)
Y = X["Survived"]
x_train, x_valdation, y_train, y_valdation = train_test_split(X_final, Y, test_size=0.2, random_state=42)


# 2. 하이퍼 파라미터 튜닝
enable_tuning = True
model = None
best_depth = 0
best_sample_leaf = 0
best_score = 0

if enable_tuning:
    for depth in range(1, 10):
        for samples_leaf in range(1, 10):
            model = DecsionTree(max_depth=depth , min_samples_leaf=samples_leaf)
            features = list(x_train.columns.values)
            model.fit(x_train, y_train, features)
            train_score = accuracy_score(model.predict(x_train), y_train)
            score = accuracy_score(model.predict(x_valdation), y_valdation)
            if best_score < score:
                best_depth, best_score = depth, score
                best_sample_leaf = samples_leaf
        print(f"max_depth: {depth} \n \t - train score: {train_score} \n \t - validation score: {score}")

    print(f"best depth: {best_depth}, best minimun leafs: {best_sample_leaf}")

    # 3. 모델 학습
    model = DecsionTree(max_depth=best_depth, min_samples_leaf=best_sample_leaf)
    features = list(x_train.columns.values)
    model.fit(x_train, y_train, features)

    # 4. 모델 정확도
    print(accuracy_score(model.predict(x_valdation), y_valdation))

else:
    model = DecsionTree(max_depth=None)
    features = list(x_train.columns.values)
    model.fit(x_train, y_train, features)

    train_score = np.mean(model.predict(x_train) == y_train)
    score = np.mean(model.predict(x_valdation) == y_valdation)
    print(f"score - train score: {train_score} \n \t - validation score: {score}")
    model.visualize_tree(model.root)

    # print(np.mean(model.predict(x_valdation) == y_valdation))


# ⭐ 모델 시각화
# model.visualize_tree(model.root)

# 학습된 모델에서 중요한 컬럼 확인
# clf = DecisionTreeClassifier(
#     criterion="gini",
#     splitter="best",
#     max_depth=best_depth,
#     min_samples_leaf=best_sample_leaf, 
#     random_state=42
# ).fit(x_train, y_train)
# print(clf.feature_importances_)
# for name, value in zip(features, clf.feature_importances_):
#     print(name, value)
# sns.barplot(x=clf.feature_importances_, y=features)
# plt.show()