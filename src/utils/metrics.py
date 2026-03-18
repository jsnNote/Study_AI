import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def accuracy_score(predict, solution):
    return np.mean(predict == solution)


# [[확률변수], [확률변수 개수], [확률]]
def _calculate_probability(data):
    RV, RV_count = np.unique(data, return_counts=True)
    n = np.sum(RV_count)

    # 0 또는 1만 가진 경우에 대한 처리
    if not 0 in RV:
        RV = np.insert(RV, 0, 0)
    elif not 1 in RV:
        RV = np.append(RV, 1)

    result = {}
    result["random_variable"] = RV
    result["random_variable_count"] = RV_count
    result["probability"] = RV_count / n

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


# model은 학습을 한 상태
class BinaryClassifierPerformance:
    def __init__(self, prediction, actual):
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self.__calculate_confusion_matrix(prediction, actual)

    def __calculate_confusion_matrix(self, prediction, actual):
        actual_np = np.array(actual)  # series를 타입변환
        # 예측 값이 postive(1)인 경우
        p_mask = prediction == 1
        # 같을 때 1, 다른 때 0
        self.TP = np.sum((prediction[p_mask] == actual_np[p_mask]))
        self.FP = np.sum(p_mask) - self.TP

        # 예측 값이 negative(0)인 경우
        # 같을 때 1, 다른 때 0
        n_mask = prediction == 0
        self.TN = np.sum(prediction[n_mask] == actual_np[n_mask])
        self.FN = np.sum(n_mask) - self.TN

    @property
    def accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

    @property
    def precision(self):
        return (self.TP) / (self.TP + self.FP)

    @property
    def recall(self):
        return (self.TP) / (self.TP + self.FN)

    @property
    def F1_score(self):
        return (2 * self.precision * self.recall) / (self.precision + self.recall)

    @property
    def one_specificity(self):
        return self.FP / (self.FP + self.TN)


class ROCCurveDrawer:
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.proba = self.model.predict_proba(x_test)[:, 1]
        self.y_true = y_test

    def calculate_roc(self):
        # 1. 확률값 정렬(이때는 y_true도 같이 정렬)
        sorted_indices = np.argsort(self.proba)[::-1]
        sorted_proba = self.proba[sorted_indices]
        sorted_y_true = np.array(self.y_true)[sorted_indices]

        # 2. 모든 확률값을 임계값 후보로 설정 & 계산
        all_thresholds = sorted_proba
        all_thresholds = np.insert(all_thresholds, 0, all_thresholds[0] + 1)
        all_thresholds = np.append(all_thresholds, 0)

        all_fpr = np.empty(len(all_thresholds))
        all_tpr = np.empty(len(all_thresholds))

        for idx, threshold in enumerate(all_thresholds):
            per = BinaryClassifierPerformance(
                (sorted_proba >= threshold), sorted_y_true
            )
            all_fpr[idx] = per.one_specificity
            all_tpr[idx] = per.recall

        fpr = []
        tpr = []
        thresholds = []
        # 3. 유의미한 지점만 남기기 (Optimization)(새로운 값을 가짐)
        # 예측 결과(FPR, TPR)가 변하지 않는 지점은 과감히 생략

        for idx, threshold in enumerate(all_thresholds):
            if (
                idx == 0
                or all_fpr[idx] != all_fpr[idx - 1]
                or all_tpr[idx] != all_tpr[idx - 1]
            ):
                fpr.append(all_fpr[idx])
                tpr.append(all_tpr[idx])
                thresholds.append(threshold)

        return fpr, tpr, thresholds

    def plot_curve(self):
        fpr, tpr, thresholds = self.calculate_roc()
        AUC = calculate_acu(fpr, tpr)
        plt.figure(figsize=(8, 6))

        plt.plot(fpr, tpr, label=f"ROC Curve (AUC: {AUC:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")  # 대각선 기준선
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend()
        plt.grid(True)
        plt.show()


def calculate_acu(fpr, tpr):
    """_summary_\n
    면적을 사다리꼴로 계산\n
    추후 적분을 이용해서 계산하는 방향으로 구현
    """

    return np.sum((np.subtract(fpr[1:], fpr[:-1]) * np.add(tpr[1:], tpr[:-1])) / 2)
