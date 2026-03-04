# Decision Tree에서의 Cross Validation

## 이론과 구현, 분석

---

# 1. Introduction(소개)

## 1.1 Motivation(동기)

교차 검증이 작은 크기 데이터에 대해서 일반적으로 좋은 성능을 낸다고 알려져 있는데 Decision Tree에 대해서도 좋을지 분석하는 것을 목표로 한다.

### Project Objectives

- 교차 검증의 이론적 정리
- cross_val_score를 통해서 Decision Tree의 성능 분석

# 2. Theoretical Background(이론적 배경)

## 2.1 Cross Validation

- 데이터를 k(하이퍼파라미터)의 부분집합으로 나눈다.
- k - 1개는 훈련용 데이터, 1개는 성능 측정용 데이터로 설정한다.
- 학습 후 성능을 구한다.
- 측정용 데이터가 겹치지 않게 하여 k번 학습을 한다.
- k개의 성능 지표의 평균을 구한다.

---

# 3. Implementation(구현)

# 3.1 의사 코드

```python

for i in range(1, k)
    train, test = split(data, i)
    tree.fit()
    scores[i] <- tree.score(test)

mean(scores)

```

---

# 4. Experimental Setup(실험 준비)

# 4.1 Datset

- titanic dataset
-

## 4.2 Hyperparameter Settings

보통 5는 계산 비용과 안정성 균형이 좋다고 알려져 있다.

- k = 5

# 5. Experimental Results(실험 결과)

|  Default   | max depth | Cross Validation |
| :--------: | :-------: | :--------------: |
|   0.7978   |     1     |      0.7834      |
|   0.7978   |     2     |      0.7736      |
|   0.8146   |   **3**   |    **0.8242**    |
| **0.8258** |   **4**   |      0.8214      |
|   0.8146   |     5     |      0.7989      |
|   0.7809   |     6     |      0.8017      |
|   0.7697   |     7     |      0.7975      |
|   0.7753   |     8     |      0.8073      |

- 성능
  - 교차 검증에서의 같은 깊이에서의 성능과 사용하지 않았을 때의 성능이 비슷함으로 보이고 있다.
  - 보통 교차검증이 성능이 더 잘 나오지만 이 실험에서는 오히려 사용하지 않았을 때의 성능이 좋았다.

# 6. Limitations(한계)

- 교차 검증의 경우 반복 학습을 통해 매개변수를 업데이트하는 것을 전제로 하고 있어 반복 학습을 하지 않고 데이터 전체를 한 번에 최적의 매개변수로 학습하는 트리에서는 기존보다 좋은 성능을 보이고 있지 않다.
- ***

# 7. Future Work(향후 연구)

- 향후 연구하는 모델에 대해서도 교차 검증을 성능 지표로 사용한다면 사용했을 때와 안 했을 때의 성능을 비교하고 선택해야 할 것 같다.

---

# 8. Conclusion

본 프로젝트에서 Cross Validation의 이론적 배경을 정리하고, Decision Tree에서의 Cross Validation 실험을 수행하였다.

이를 통해

- 교차 검증의 원리에 대한 이해
- 교차 검증이 유용한 지표인지 확인
- Decision Tree에서의 교차 검증의 유용성

을 수행하였다.
