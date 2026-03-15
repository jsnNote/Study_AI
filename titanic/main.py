from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

src = "../datasets/titanic/"

# 0. 목표 설정
#   - 타이타닉 침몰 사고에서 승객이 살아남았는지 여부 예측
#   - 출력 (PassengerId, 생존여부)
# 1. 데이터 분석
# analysis.ipynb
# 2. 데이터 처리
train = pd.read_csv(src + "train.csv")

# 인코딩
# male - 0, female - 1
train["Sex_numeric"] = train["Sex"].map({"male": 0, "female": 1})

#   - 데이터 가공
# print(train["Age"].isnull().sum()) # Age는 177개의 결측치가 존재 -> 결측치 처리

train["Title"] = train["Name"].str.extract(" ([A-Za-z]+)\.")
median_ages = train.groupby("Title")["Age"].transform("median")
train["Age"] = train["Age"].fillna(median_ages)

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

train = train.dropna(axis=1)
train = train.select_dtypes(exclude=["str"])

X = train.drop("Survived", axis=1)
Y = train["Survived"].values

# 특성 선택
used_SelectKBest = False
feat_sel = None

if used_SelectKBest:
    #   - SelectKBest ✅
    feat_sel = SelectKBest(f_classif, k=3)
    X_selected = feat_sel.fit_transform(X, Y)
else:
    #   - SelectKBest ❌
    # print(train.isnull().sum())  # 각 특성의 결측치 확인
    feat_sel = ["Pclass", "Sex_numeric", "Age"]
    X_selected = X[feat_sel]

    # X_selected = X[["Pclass"]] # Accurancy - 0.68
    # X_selected = X[["Sex_numeric"]]  # Accurancy - 0.73
    # X_selected = X[["Age"]]  # Accurancy - 0.68
    # X_selected = X[["FamilySize"]]  # Accurancy - 0.68
    # X_selected = X[["Sex", "Pclass"]]  # Accurancy - 0.68

    X_selected = X[["Pclass", "Sex_numeric", "Age"]]  # Accurancy - 0.73
# X_selected = X[["Pclass", "Sex", "Age", "FamilySize"]]  # Accurancy - 0.74
# => 정확도가 비슷한 경우에는 특성을 줄여서 모델의 복잡도를 줄여야 함(오캄의 면도날).
# - 장점 : 과적합 방지, 모델의 해석력 향상(특성이 적으면 결과가 어떻게 나왔는지 분석이 쉬움), 차원의 저우를 줄일 수 있음.

minmax_sclaer = MinMaxScaler()  # 특성의 범위가 불균형해서 0~1 범위로 스케일링
minmax_sclaer.fit(X_selected)
X_scaled = minmax_sclaer.transform(X_selected)

x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)


# 3. 모델 구성
class BatchLogistic:
    def __init__(self, learning_rate=0.01):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.losses = []

    def forpass(self, x):
        y_hat = np.dot(x, self.w) + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = np.dot(x.T, err) / x.shape[0]
        b_grad = np.mean(err)
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, epochs=1000):
        self.w = np.zeros(x.shape[1])
        self.b = 0
        for i in range(epochs):
            z = self.forpass(x)
            y_hat = self.activation(z)
            y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
            loss = -np.mean(
                y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped)
            )

            err = y_hat - y
            w_grad, b_grad = self.backprop(x, err)
            self.w -= self.lr * w_grad
            self.b -= self.lr * b_grad

            self.losses.append(loss)
            if (i + 1) % 100 == 0:
                print(f"Epoch: {i + 1}/{epochs} - Loss: {loss}")

    def predict(self, x):
        z = self.forpass(x)
        a = self.activation(z)
        return a > 0.5


class MiniBatchLogistic:
    def __init__(self, learning_rate=0.01):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.losses = []

    def forpass(self, x):
        y_hat = np.dot(x, self.w) + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = np.dot(x.T, err) / x.shape[0]
        b_grad = np.mean(err)
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, epochs=1000, batch_size=30):
        self.w = np.zeros(x.shape[1])
        self.b = 0
        for i in range(epochs):
            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            total_loss = 0
            for j in range(0, x.shape[0], batch_size):
                x_batch = x_shuffled[j : j + batch_size]
                y_batch = y_shuffled[j : j + batch_size]

                z = self.forpass(x_batch)
                y_hat = self.activation(z)
                y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)

                loss = -np.mean(
                    y_batch * np.log(y_hat_clipped)
                    + (1 - y_batch) * np.log(1 - y_hat_clipped)
                )

                err = y_hat - y_batch
                w_grad, b_grad = self.backprop(x_batch, err)
                self.w -= self.lr * w_grad
                self.b -= self.lr * b_grad

                total_loss += loss

            num_batches = np.ceil(x.shape[0] / batch_size)
            self.losses.append(total_loss / num_batches)

            if (i + 1) % 100 == 0:
                print(f"Epoch: {i + 1}/{epochs} - Loss: {self.losses[-1]}")

    def predict(self, x):
        z = self.forpass(x)
        a = self.activation(z)
        return a > 0.5


class StochasticLogisic:  # stochastic
    def __init__(self, learning_rate=0.01):
        self.w = None
        self.b = None
        self.lr = learning_rate
        self.losses = []

    def forpass(self, x):
        y_hat = np.dot(x, self.w) + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = err * x
        b_grad = 1 * err
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            total_loss = 0
            for x_i, y_i in zip(x, y):
                z = self.forpass(x_i)
                y_hat = self.activation(z)
                loss = -np.mean(y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat))
                err = y_hat - y_i
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= self.lr * w_grad
                self.b -= self.lr * b_grad

                total_loss += loss
            self.losses.append(total_loss / x.shape[0])
            if (i + 1) % 100 == 0:
                print(f"Epoch: {i + 1}/{epochs} - Loss: {self.losses[-1]}")

    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(z)
        return a > 0.5


# 4. 모델 실행
Smodel = StochasticLogisic()
Mmodel = MiniBatchLogistic()
Bmodel = BatchLogistic()

Smodel.fit(x_train, y_train, 1000)
Mmodel.fit(x_train, y_train, 1000)
Bmodel.fit(x_train, y_train, 1000)

print(np.mean(Smodel.predict(x_train) == y_train))
print(np.mean(Mmodel.predict(x_train) == y_train))
print(np.mean(Bmodel.predict(x_train) == y_train))

plt.plot(Smodel.losses, color="red", label="Stochastic")
plt.plot(Mmodel.losses, color="blue", label="MiniBatch")
plt.plot(Bmodel.losses, color="green", label="Batch")
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# 5. 추론

test = pd.read_csv(src + "test.csv")
#   - 데이터 가공
test["Sex_numeric"] = test["Sex"].map({"male": 0, "female": 1})
test["Title"] = test["Name"].str.extract(" ([A-Za-z]+)\.")
median_ages = test.groupby("Title")["Age"].transform("median")

test["Age"] = test["Age"].fillna(median_ages)
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
test = test.dropna(axis=0)
test = test.dropna(axis=1)
test = test.select_dtypes(exclude=["str"])
X_test = test

if used_SelectKBest:
    #   - SelectKBest ✅
    X_test_selected = feat_sel.fit_transform(X_test)
else:
    #   - SelectKBest ❌
    X_test_selected = X_test[feat_sel]

minmax_sclaer = MinMaxScaler()  # 특성의 범위가 불균형해서 0~1 범위로 스케일링
minmax_sclaer.fit(X_test_selected)
X_test_scaled = minmax_sclaer.transform(X_test_selected)

for i, survived in zip(
    range(X_test.shape[0]),
    list(map({True: 1, False: 0}.get, Mmodel.predict(X_test_scaled))),
):
    print(X_test["PassengerId"].iloc[i], survived)

# 로지스틱 회귀는 모델이 단순하기 때문에 모델의 복잡함보다는 데이터에 따라서 성능이 달라짐
# 방법
# 1. 데이터의 특성을 조합
#   - SibSp + Parch + 본인 - FamilySize로 데이터를 조합
#   - 정확도가 약 1% 올라감.

# 2. 학습할 때 배치의 데이터 크기를 줄임(미니 배치 방식).
#   - 배치 정확도 - 약 73%
#   - 미니 배치 정확도 - 약 75%
#   - stochastic 정확도 - 약 78%
# 속도1위: stochastic, 안정성 1위: 배치, 균형 1위: 미니 배치
# 결측치 처리 방식 -> 결측치가 있는 데이터 제거

# 3. 데이터의 결측치 처리
# - Age의 경우 결측치가 177개 정도 있는데 이를 Mr, Mrs, Miss 등의 나이를 짐작할 수 있는 단어를 통해서 그룹에 중앙값으로 결측치를 채움
# - 이후 성능이 모든 모델이 78 ~ 79% 정도가 됨.
# - ⭐ 데이터의 중요성을 알게 됨.
