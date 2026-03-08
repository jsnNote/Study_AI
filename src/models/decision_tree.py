import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from ..utils.tools import find_best_split, mode
class Node:
    def __init__(self, left=None, right=None, threshold=None, feature=None, is_leaf=False, value=None, depth=0):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.is_leaf = is_leaf
        self.value = value
        self.depth = depth
class DecsionTree:
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
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return Node(is_leaf=True, value=mode(y), depth=depth)
        
        # 5. feature가 없는 경우
        elif len(features) == 0 or best_split["feature"] == None: 
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth) # leaf node로 만들면 됨.
        
        root = Node(**best_split, depth=depth)
        remaining_features = [f for f in features if f != best_split["feature"]]
        # print(f"Node -feature: {root.feature}, depth: {root.depth}, (flag:{flag}) {len(remaining_features)}")
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
            return f"Predict: {int(node.value)}, Depth: {node.depth}"
        
        return {
        f"{node.feature} <= {float(node.threshold):.2f}": {
            "left": self.structed_tree(node.left),
            "right": self.structed_tree(node.right),
            "depth": node.depth
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
    


# 하위 노드에서도 상위 노드의 특징은 사용 가능(연속형 데이터의 경우)
# 위에 DecisionTree는 한 번 사용한 특징은 사용 못하게 했음에도 성능이 좋았던 것은 특징 선택을 잘해서 특징 간의 분별력이 있었기 때문임.
class RandomForestTree:
    def __init__(self, max_features, max_depth=4, min_samples_leaf=5, features=None, random_state=None):
        self.root = None
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.features = features
        self.random_state=random_state

    def _build_tree(self, X, y, features, depth=0, flag="root"):
        best_split = find_best_split(X[features], y, features)

        # 1. 데이터가 없을 때 -> y가 없냐
        if y.empty:
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth) 
        
        # 2. 데이터 개수가 적은 경우에는 반환
        elif X.shape[0] < self.min_samples_leaf:
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth)
        
        # 3. 대상 속성이 단일값을 가진 경우 -> label이 단일값인지 보면 됨.
        elif len(np.unique(y)) == 1: 
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth)
    
        # 4. 깊이 제한
        elif self.max_depth is not None and depth >= self.max_depth:
            return Node(is_leaf=True, value=mode(y), depth=depth)
        
        # 5. feature가 없는 경우
        elif len(features) == 0 or best_split["feature"] == None: 
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth)
        
        root = Node(**best_split, depth=depth)

        left_idx = np.where(X[best_split["feature"]] <= best_split["threshold"])[0]
        right_idx = np.where(X[best_split["feature"]] > best_split["threshold"])[0]

        #  6 threhold에 한 쪽만 값이 있는 경우 -> 깊이가 더 깊어지는 문제가 발생
        if (len(left_idx) == 0 or len(right_idx) == 0):
            # print(f"leaf - depth: {depth}, (flag:{flag}) value: {mode(y)}")
            return  Node(is_leaf=True, value=mode(y), depth=depth)
        
        rng = np.random.default_rng(seed=self.random_state)

        left_feature_subset = rng.choice(self.features, size=self.max_features, replace=False)
        left_subset_X = X.iloc[left_idx]
        left_subset_y = y.iloc[left_idx]
        root.left = self._build_tree(left_subset_X, left_subset_y, left_feature_subset, depth=(depth+1), flag="left")

        right_feature_subset = rng.choice(self.features, size=self.max_features, replace=False)
        right_subset_X = X.iloc[right_idx]
        right_subset_y = y.iloc[right_idx]
        root.right = self._build_tree(right_subset_X, right_subset_y, right_feature_subset, depth=(depth + 1), flag="right")
        
        return root
    
    def structed_tree(self, node):
        if node is None:
            return None

        if node.is_leaf:
            return f"Predict: {int(node.value)}, Depth: {node.depth}"
        
        return {
        f"{node.feature} <= {float(node.threshold):.2f}": {
            "left": self.structed_tree(node.left),
            "right": self.structed_tree(node.right),
            "depth": node.depth
        }
    }
    
    def visualize_tree(self, node):
        print(json.dumps(self.structed_tree(node), indent=4, ensure_ascii=False))
    


    def fit(self, X, y):
        rng = np.random.default_rng(seed=self.random_state)
        fetures_subset = rng.choice(self.features, size=self.max_features, replace=False)
        self.root = self._build_tree(X, y, fetures_subset)

    def _traverse_tree(self, X, node):
        if node.is_leaf:
            return node.value
        
        if X[node.feature] <= node.threshold:
            return self._traverse_tree(X, node.left)
        else:
            return self._traverse_tree(X, node.right)
        
    def predict(self, X):
        return X.apply(lambda row: self._traverse_tree(row, self.root), axis=1).values