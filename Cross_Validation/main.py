import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


src = "../datasets/titanic/"

train = pd.read_csv(src + "train.csv")

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

x_train, x_val, y_train, y_val = train_test_split(
    X_final, Y, test_size=0.2, random_state=42
)

for depth in range(1, 10):
    dt_clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=depth,
        random_state=42
    ).fit(x_train, y_train)

    score = np.round(np.mean(dt_clf.predict(x_val) == y_val), 4)
    print(f"Default (depth:{depth})\n - scores: {score}")


    cross_dt_clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=depth,
        random_state=42
    ).fit(x_train, y_train)

    scores = cross_val_score(cross_dt_clf, x_train, y_train, scoring="accuracy", cv=5)
    mean = np.round(np.mean(scores), 4)
    print(f"Cross Validation (depth: {depth})\n - scores: {mean}")