import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from sklearn.model_selection import train_test_split

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

# 범주형 -> 연속형 데이터로 만듦(처리가 쉬움. 단, 과적합에 취약)
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
x_train, x_valdation, y_train, y_valdation = train_test_split(X_final, Y, test_size=0.2, random_state=42)


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


# find_best_split - 연속 속성에서 최고의 분할을 찾 -> 하나의 속성에 대해서도 동작하게 수정
def find_best_split(X, y, features=None):
    # 모든 입력을 연속 속성으로 하고 있음.
    # 정렬 후 두 데이터의 중간값을 기준으로 지니계수를 구함.
    best_split_gini = np.inf
    best_threshold = -1
    best_feature = None
    split_info = {"feature": [], "threshold": []}
    X_np = np.array(X)
    y_np = np.array(y)
    X_columns = X.columns.values
    for feature_idx in range(X.shape[1]):
        column_values = X_np[:, feature_idx]
        idx_sorted = np.argsort(column_values)  # x가 DataFrame이면 안됨
        x_sorted = column_values[idx_sorted]
        y_sorted = y_np[idx_sorted]
        v = np.unique(np.sort(x_sorted))
        thresholds = (v[:-1] + v[1:]) / 2
        # feature = (
        #     features[feature_idx] if features else X_columns[feature_idx]
        # )  # else DataFrame으로 들어올 때, if는 그 외의 경우
        feature = features[feature_idx] if features else X_columns[feature_idx]
          # else DataFrame으로 들어올 때, if는 그 외의 경우
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

# series의 경우 [[column]]으로 가져옴
# print(find_best_split(X_final[:][["Pclass", "Age"]], Y))
# print(np.unique([1, 1, 1, 0]).shape[0] == 1)
# print(X_final[:][X_final.columns[:3]])

def mode(data):
    if len(data) == 0:
        return None
    
    values, counts = np.unique(data, return_counts=True)
    max_idx = np.argmax(counts)

    return values[max_idx]
class Node:
    def __init__(self, left=None, right=None, threshold=None, feature=None, is_leaf=False, value=None, depth=0):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.is_leaf = is_leaf
        self.value = value
        self.depth = depth
        
class Decsion_Tree:
    def __init__(self, max_depth=4, min_samples_leaf=5):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # _build_tree - 속성 중에서 분할 할 속성을 찾고 분할 후 서브 트리를 통해서 재귀적으로 트리 구축
    # 모든 특성(Feature)을 다 쓰는 것"이 아니라, 데이터를 가장 잘 나누는 "최적의 특성"을 매 순간 골라내는 것
    def _build_tree(self, X, y, features, depth=0, flag="root"):  # 수치형 feature로 처리
        best_split = find_best_split(X, y, features)

        # 중지 기준
        # 잎 노드는 value만 있고, feature는 없음
        # feature 설정해야함.
        # 1. 데이터가 없을 때 -> y가 없냐
        if y.empty:
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth) # leaf node로 만들면 됨.
        
        # 2. 데이터 개수가 적은 경우에는 반환
        elif X.shape[0] < self.min_samples_leaf:
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth) # leaf node로 만들면 됨.
        
        # 3. 대상 속성이 단일값을 가진 경우 -> label이 단일값인지 보면 됨.
        elif len(np.unique(y)) == 1: 
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth) # leaf node로 만들면 됨.
    
        # 4. 깊이 제한
        elif self.max_depth is not None and depth >= self.max_depth:
            return Node(is_leaf=True, value=mode(y), depth=depth)
        
        # 5. feature가 없는 경우
        elif len(features) == 0 or best_split["feature"] == None: 
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth) # leaf node로 만들면 됨.
        
        root = Node(**best_split, depth=depth)
        remaining_features = [f for f in features if f != best_split["feature"]]
        # print(f"Node -feature: {root.feature}, depth: {root.depth}, (flag:{flag}) {len(remaining_features)}")
        # sub_tree
        # subset을 만들어야 함.
        # 분할하는 feature에 대한 인덱스를 구해야 함.
        left_idx = np.where(X[best_split["feature"]] <= best_split["threshold"])[0]
        right_idx = np.where(X[best_split["feature"]] > best_split["threshold"])[0]

        #  6 threhold에 한 쪽만 값이 있는 경우 -> 깊이가 더 깊어지는 문제가 발생
        if (len(left_idx) == 0 or len(right_idx) == 0):
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth)
         
        # 형제 노드는 같은 특성을 가지고 있을 수 있음. 왜냐하면 분할 이후에 threshold에 따라서 데이터가 분할되고 그 데이터만으로 분할이 또 일어남(독립적)
        left_subset_X = X.iloc[left_idx][remaining_features]
        left_subset_y = y.iloc[left_idx]
        root.left = self._build_tree(left_subset_X, left_subset_y, remaining_features, depth=(depth+1), flag="left")

        right_subset_X = X.iloc[right_idx][remaining_features]
        right_subset_y = y.iloc[right_idx]
        root.right = self._build_tree(right_subset_X, right_subset_y, remaining_features, depth=(depth + 1), flag="right")
        
        return root
    
    def structed_tree(self, node):
        if node is None:
            return None

        if node.is_leaf:
            return f"Predict({node.depth}): {int(node.value)}"
        
        return {
        f"{node.feature} <= {float(node.threshold):.2f} ({node.depth})": {
            "left": self.structed_tree(node.left),
            "right": self.structed_tree(node.right)
        }
    }
    
    def visualize_tree(self, node):
        print(json.dumps(self.structed_tree(node), indent=4, ensure_ascii=False))
    


    def fit(self, X, y, features):
        self.root = self._build_tree(X, y, features)

    def _traverse_tree(self, X, node):
        if node.is_leaf:
            return node.value
        
        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)
        else:
            return self._traverse_tree(X, node.right)
        
    def predict(self, X):
        return X.apply(lambda row: self._traverse_tree(row, self.root), axis=1).values

enable_tuning = False
model = None
if enable_tuning:
    best_depth = 0
    best_sample_leaf = 0
    best_score = 0
    for depth in range(1, 10):
        for samples_leaf in range(1, 10):
            model = Decsion_Tree(max_depth=depth)
            features = list(x_train.columns.values)
            model.fit(x_train, y_train, features)
            score = np.mean(model.predict(x_valdation) == y_valdation)
            if best_score < score:
                best_depth, best_score = depth, score
                best_sample_leaf = samples_leaf

    print(f"best depth: {best_depth}, best # minimun leafs: {best_sample_leaf}")

    model = Decsion_Tree(max_depth=best_depth, min_samples_leaf=best_sample_leaf)

model = Decsion_Tree()
features = list(x_train.columns.values)
model.fit(x_train, y_train, features)
print(np.mean(model.predict(x_valdation) == y_valdation))

model.visualize_tree(model.root)
