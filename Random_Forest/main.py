import sys, os, time

# 현재 파일(main.py)의 부모(A)의 부모(my_ml_project) 경로를 가져옴
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 파이썬 모듈 검색 경로 최상단에 추가
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import train_test_split
from src import RandomForest, ParalleRandomForest
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
features = X_final.columns.values
Y = X["Survived"]
x_train, x_validation, y_train, y_validation = train_test_split(X_final, Y, test_size=0.2, random_state=42)



def time_test(n_estimators):
    
    rf = RandomForest(n_estimators, max_depth=4, min_samples_leaf=1)
    start_time = time.time()
    rf.fit(x_train, y_train, features)
    end_time = time.time()
    rf_time = end_time - start_time
    
    prf = ParalleRandomForest(100, max_depth=4, min_samples_leaf=1)
    start_time = time.time()
    prf.fit(x_train, y_train, features)
    end_time = time.time()
    prf_time = end_time - start_time

    return (rf_time, prf_time)


# Process를 만드는 거는 무거운 작업이기 때문에 작업이 적은 경우에는 오히려 순차적인 처리가 빠르다. -> 작업이 많으면 오히려 빠름
if __name__ == '__main__':
    # # 순차 처리 랜덤포레스트
    # rf = RandomForest(100, max_depth=4, min_samples_leaf=1)
    # rf.fit(x_train, y_train, features)
    # print(rf.score(x_valdation, y_valdation))
    
    # # 병렬 처리 랜덤포레스트
    # prf = ParalleRandomForest(100, max_depth=4, min_samples_leaf=1)
    # prf.fit(x_train, y_train, features)
    # print(prf.score(x_valdation, y_valdation))
    
    # 순차 처리 vs 병렬 처리 시간 테스트 코드
    for n in range(100, 501, 100):
        rf_t, prf_t = time_test(n)
        print(f"Number of trees [{n}] \n - Sequential Random Forest Traing Time: {rf_t:.4f} \n - Paralle Random Forest Traing Time: {prf_t:.4f}")


    # # oob_err test code
    # rf = RandomForest(100, max_depth=4, min_samples_leaf=1)
    # rf.fit(x_train, y_train, features)
    # print(rf.oob_err)
    
    # prf = ParalleRandomForest(100, max_depth=4, min_samples_leaf=1)
    # prf.fit(x_train, y_train, features)
    # print(prf.oob_err)
    
    