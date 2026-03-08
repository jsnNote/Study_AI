import numpy as np
import pandas as pd
from pandas import DataFrame

from .decision_tree import RandomForestTree
from ..utils.tools import mode

class RandomForest: # max_features은 보통 sqrt(max_features) 사용
    def __init__(self, n_estimators, max_features=None, max_depth=4, min_samples_leaf=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = None
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = None

    def _build_tree(self, X, y, features=None):
        if not features:
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
            subset_x, subset_y = self.__bootstrap_sample(X, y, i)
            
            tree.fit(subset_x, subset_y)

    def __bootstrap_sample(self, X, y, i):
        rng = np.random.default_rng(seed=self.random_state+i)
        idx = rng.choice(X.index, size=self.max_samples)
        bootstrap_x = X.loc[idx]
        bootstrap_y = y.loc[idx]
        return bootstrap_x, bootstrap_y

    def fit(self, X, y, features=None):
        self.max_samples = np.ceil((X.shape[0] * 0.632)).astype(dtype=np.int64)
        self.max_features = np.ceil(np.sqrt(len(X.columns))).astype(dtype=np.int64) if not self.max_features else self.max_features
        self._build_tree(X, y, features)

    def predict(self, X):
        categories = []
        for tree in self.trees:
            categories.append(tree.predict(X))

        categories = np.array(categories)
        result = np.apply_along_axis(mode, axis=0, arr=categories)
        return result


    def score(self, X, y):
        return np.mean(self.predict(X) == y)