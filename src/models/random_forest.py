import numpy as np
import pandas as pd
from pandas import DataFrame
from .decision_tree import RandomForestTree
from ..utils.tools import mode, accuracy_score

from multiprocessing import Pool, current_process, Manager

# 구현 특징
# - 트리마다 입력 X를 부트스트래핑을 통해서 구한다.
# - 분할 할 때마다 특징의 subset을 구한다.

# 최적화
# - pandas 객체 -> numpy 객체
# - column을 스트링 그대로가 아니라 index로 다뤄야함.
class RandomForest: # max_features은 보통 sqrt(max_features) 사용
    def __init__(self, n_estimators, max_features=None, max_depth=4, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = None
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = None
        self.oob_scores = []
        
    def _build_tree(self, X, y, features=None):
        if features is None:
            features = X.columns.values

        self.trees = [
            RandomForestTree(
                max_features=self.max_features, 
                max_depth=self.max_depth, 
                min_samples_leaf=self.min_samples_leaf, 
                features=features, 
                random_state=(self.random_state + i) 
                ) for i in range(self.n_estimators)]  # 재현성을 위해서 random_state 추가
        
        for i, tree in enumerate(self.trees):
            # bootstrap
            subset_x, subset_y, oob_x, oob_y = self.__bootstrap_sample(X, y, i)
            
            tree.fit(subset_x, subset_y)
            self.oob_scores.append(accuracy_score(tree.predict(oob_x), oob_y))

    def __bootstrap_sample(self, X, y, i):
        rng = np.random.default_rng(seed=self.random_state+i)
        idx = rng.choice(X.index, size=self.max_samples, replace=True)
        bootstrap_x = X.loc[idx]
        bootstrap_y = y.loc[idx]
        oob_idx = X.index.difference(idx)
        oob_x = X.loc[oob_idx]
        oob_y = y.loc[oob_idx]
        
        return bootstrap_x, bootstrap_y, oob_x, oob_y

    def fit(self, X, y, features=None):
        self.max_samples = np.ceil((X.shape[0] * 0.632)).astype(dtype=np.int64) #Out-of-Bag(OOB)를 근거로 설명하기
        if self.max_features is None:
            self.max_features = np.ceil(np.sqrt(len(X.columns))).astype(dtype=np.int64) 
        self._build_tree(X, y, features)

    def predict(self, X):
        categories = np.array([tree.predict(X) for tree in self.trees])
        
        # axis=0(열 단위)로 categories에 대해서 mode()를 적용한다.
        result = np.apply_along_axis(mode, axis=0, arr=categories)
        return result


    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    @property
    def oob_accuracy(self):
        return np.mean(self.oob_scores)
        


class ParallelRandomForest:
    def __init__(self, n_estimators, max_features=None, max_depth=4, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = None
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = None
        self.oob_scores = Manager().list()

    def __bootstrap_sample(self, X, y, i):
        rng = np.random.default_rng(seed=self.random_state+i)
        idx = rng.choice(X.index, size=self.max_samples, replace=True)
        bootstrap_x = X.loc[idx] 
        bootstrap_y = y.loc[idx]
        oob_idx = X.index.difference(idx)
        oob_x = X.loc[oob_idx]
        oob_y = y.loc[oob_idx]
        
        return (bootstrap_x, bootstrap_y, oob_x, oob_y)
    
    def _build_tree(self, tree_info): 
        dt, X, y, i, oob_scores = tree_info
        subset_x, subset_y, oob_x, oob_y = self.__bootstrap_sample(X, y, i)
        dt.fit(subset_x, subset_y)
        oob_scores.append(accuracy_score(dt.predict(oob_x), oob_y))
        
        return dt
        
        
    def fit(self, X, y, features=None):
        n_cpu = 8
        self.max_samples = np.ceil((X.shape[0] * 0.632)).astype(dtype=np.int64) #Out-of-Bag(OOB)를 근거로 설명하기 OOB 검증만들기
        if self.max_features is None:
            self.max_features = np.ceil(np.sqrt(len(X.columns))).astype(dtype=np.int64)
        trees = [
            RandomForestTree(
                max_features=self.max_features, 
                max_depth=self.max_depth, 
                min_samples_leaf=self.min_samples_leaf, 
                features=features, 
                random_state=(self.random_state + i) 
                ) for i in range(self.n_estimators)]
        
        tasks = [(tree, X, y, i, self.oob_scores) for i, tree in enumerate(trees)]

        with Pool(processes=n_cpu) as p:
            self.trees = p.map(self._build_tree, tasks)
            
            
    def predict(self, X):
        categories = np.array([tree.predict(X) for tree in self.trees])
        
        # axis=0(열 단위)로 categories에 대해서 mode()를 적용한다.
        result = np.apply_along_axis(mode, axis=0, arr=categories)
        return result


    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    @property
    def oob_accuracy(self):
        return np.mean(self.oob_scores)