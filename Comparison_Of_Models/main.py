# Logistic Regression, KNN, Decision Tree, Random Forest 모델 성능 비교
import sys, os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
    
from src.utils.metrics import BinaryClassifierPerformance
from src import DecsionTree, KNN, ParallelRandomForest, BatchLogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pandas import DataFrame
from src import categorical_data_preprocessing, one_hot_encoding, accuracy_score
import src.core.distance as distance

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

drop_index = train["Embarked"].dropna(axis=0).index
train_final = train.iloc[drop_index]

embarked_encoding = one_hot_encoding(train_final["Embarked"], "Embarked")
X = categorical_data_preprocessing(train_final, embarked_encoding, feature="Embarked")
X_final = data_preprocessing(X).drop(labels="Survived", axis=1)
Y = X["Survived"]
x_train, x_validation, y_train, y_validation = train_test_split(X_final, Y, test_size=0.2, random_state=42)
features = list(x_train.columns.values)

# Decision Tree
dt = DecsionTree(max_depth=4, min_samples_leaf=1)
dt.fit(x_train, y_train, features)
dt_per = BinaryClassifierPerformance(dt.predict(x_validation), y_validation)
print("Decision Tree")
print(f"Accuracy: {dt_per.accuracy}")
print(f"Precision: {dt_per.precision}")
print(f"Recall: {dt_per.recall}")
print(f"F1-score: {dt_per.F1_score}")

# Random Forest
prf = ParallelRandomForest(n_estimators=500, max_depth=4, min_samples_leaf=1)
prf.fit(x_train, y_train, features)
rf_per = BinaryClassifierPerformance(prf.predict(x_validation), y_validation)
print("Random Forest")
print(f"Accuracy: {rf_per.accuracy}")
print(f"Precision: {rf_per.precision}")
print(f"Recall: {rf_per.recall}")
print(f"F1-score: {rf_per.F1_score}")

# Batch Logistic Regression
blr = BatchLogisticRegression()
blr.fit(np.array(x_train['Sex_numeric']).reshape(x_train.shape[0], 1), y_train, 300)
blr_per = BinaryClassifierPerformance(blr.predict(np.array(x_validation['Sex_numeric'])).reshape(x_validation.shape[0]), y_validation)
print("Batch Logistic Regression")
print(f"Accuracy: {blr_per.accuracy}")
print(f"Precision: {blr_per.precision}")
print(f"Recall: {blr_per.recall}")
print(f"F1-score: {blr_per.F1_score}")