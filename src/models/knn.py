import numpy as np
import pandas as pd
from pandas import DataFrame
from ..core.distance import minkowski, manhattan, euclidean


class KNN:
    def __init__(self, k: int = 3, metric=minkowski, p: int = 2):
        self.k = k  # 짝수이면 동점이 발생해서 확률이 모호해짐 -> 홀수로 설정
        self.metric = None
        self.p = p
        self.X = None
        allowed_metrics = [manhattan, minkowski, euclidean]
        if metric in allowed_metrics:
            self.metric = metric
        else:
            raise ValueError(
                "metric은 manhattan, minkowski, euclidean 중 하나여야 합니다."
            )

    def _get_distance(self, x1, x2):
        if self.metric == minkowski:
            return self.metric(x1, x2, self.p)
        return self.metric(x1, x2)

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, data: np.ndarray):
        if data.ndim == 1:
            data = data.reshape((1, -1))

        size = data.shape[0]
        predict_value = np.empty(size, dtype=int)

        for item, idx in zip(data, range(size)):
            distance = self._get_distance(self.X_train, item)
            distance_list = [(d, l) for d, l in zip(distance, self.Y_train)]
            distance_list.sort(key=lambda x: x[0])
            k_label_list = distance_list[: self.k]
            label_cnt = dict.fromkeys(set([v[1] for v in k_label_list]), 0)

            for _, label in k_label_list:
                label_cnt[label] += 1

            for k, v in label_cnt.items():
                if max(label_cnt.values()) == v:
                    predict_value[idx] = k

        return predict_value

    def predict_proba(self, data: np.ndarray):
        if data.ndim == 1:
            data = data.reshape((1, -1))

        size = data.shape[0]
        predict_proba_list = np.empty((size, 2))

        for idx, item in enumerate(data):
            distance = self._get_distance(self.X_train, item)
            distance_list = np.column_stack(
                (distance, self.Y_train)
            )  # distance는 정렬을 위한 기준
            sorted_idx = np.argsort(distance_list[:, 0])
            sorted_distance = distance_list[sorted_idx]
            k_label = sorted_distance[: self.k, 1]

            # 0 또는 1이 없는 경우가 존재
            if not 0 in k_label:
                prob = np.array((0, 1)).reshape(1, -1)
            elif not 1 in k_label:
                prob = np.array((1, 0)).reshape(1, -1)
            else:
                prob = (np.unique(k_label, return_counts=True)[1] / self.k).reshape(
                    1, -1
                )
            predict_proba_list[idx] = prob

        return predict_proba_list
