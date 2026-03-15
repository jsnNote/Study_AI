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

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from src import one_hot_encoding, categorical_data_preprocessing

path = "../datasets/titanic/"

train = pd.read_csv(path + "train.csv")

# 데이터 전처리
def data_preprocessing(df: DataFrame):
    new_df = df.copy()
    new_df["Sex_numeric"] = new_df["Sex"].map({"male": 0, "female": 1})
    new_df["Title"] = new_df["Name"].str.extract(" ([A-Za-z]+)\.")
    median_ages = new_df.groupby("Title")["Age"].transform("median")
    new_df["Age"] = new_df["Age"].fillna(median_ages)
    new_df["FamilySize"] = new_df["SibSp"] + new_df["Parch"] + 1

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


drop_index = train["Embarked"].dropna(axis=0).index
train_final = train.iloc[drop_index]
embarked_encoding = one_hot_encoding(train_final["Embarked"], "Embarked")
X = categorical_data_preprocessing(train_final, embarked_encoding, feature="Embarked")
X_final = data_preprocessing(X).drop(labels="Survived", axis=1)
Y = X["Survived"]

x_train, x_val, y_train, y_val = train_test_split(
    X_final, Y, test_size=0.2, random_state=42
)

for depth in range(1, 10):
    print(f"[ Depth: {depth} ]".center(30, "="))

    dt_clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=depth,
        random_state=42
    ).fit(x_train, y_train)

    score = np.round(np.mean(dt_clf.predict(x_val) == y_val), 4)
    print(f"Default     score: {score}")


    cross_dt_clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=depth,
        random_state=42
    ).fit(x_train, y_train)

    scores = cross_val_score(cross_dt_clf, x_train, y_train, scoring="accuracy", cv=5)
    mean = np.round(np.mean(scores), 4)
    print(f"Cross Validation score: {mean}")
