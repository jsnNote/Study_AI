# 🌳 Decision Tree from Scratch

## 이론과 구현, 분석

---

# 1. Introduction(소개)

## 1.1 Motivation(동기)

Decsion Tree는 입력을 재귀적으로 분할하여 분류 규칙을 찾아 상수 예측을 수행하는 모델이다.

본 프로젝트는 라이브러리 사용을 넘어서 수학적 원리의 이해를 통해서 직접 구현을 목표로 한다.

### Project Objectives

- Decision Tree의 이론적 배경 정리
- Gini 기반 CART 알고리즘 구현
- sklearn 구현과의 비교

# 2. Theoretical Background(이론적 배경)

## 2.1 Entrophy(Information Theory)

섀넌의 <통신의 수학적 이론>에서 정의된 평균 정보량(엔트로피)

정보량

$$
I(x) = -\log_2p(x)
$$

엔트로피

$$
H(X) = \sum_{i=1}^{n}p(x_i)I(x_i) =  -\sum_{i=1}^{n}p(x_i)\log_{2}p(x_i)
$$

- 확률이 균등할 경우(균등 분포) -> 엔트로피가 최대
- 하나의 클래스인 경우 -> 엔트로피가 0

---

## 2.2 Gini

GINI는 데이터의 불순도를 측정하는 지표

$$
GINI(t) = 1 - \sum_{j=1}^{k}[p(j|t)]^2
$$

- 단일값을 가지는 경우 -> 0
- 균등한 경우 -> 0.5(최대)

---

## 2.3 Informatio Gain

분할을 했을 때 엔트로피(정보량)의 감소량를 측정하는 지표

정보획득 값이 클수록 불확실성이 많이 제거됨을 나타낸다.

$$Information \space GAIN = (전체 엔트로피) - (분할 후 평균 엔트로피)$$
$$Information \space GAIN = Entropy(p) - \sum_{i=1}^{k}\frac{n_i}{n}Entropy(i)$$

## 2.4 CART 알고리즘

- 속성별로 최적의 분할을 찾는다.
  - 최적의 분할은 Gini, Entrophy가 가장 많이 감소하는 지점
- 이진 분류
- 연속형 feature 지원

---

# 3. Implementation(구현)

## 3.1 분할 기준

- 값 정렬
- 인접한 두 값의 중간값을 임계값으로 지정
- 임계값을 기준으로 Split Gini 계산
- 최소값으로 분할 선택

---

## 3.2 정지 조건

- 데이터가 없는 경우
- 최소 샘플 수
- 속성이 순수할 때
- 최대 깊이
- 분할 속성없어 더 이상 분할 불가능
- 임계값에 대해서 한 쪽만 값을 가지는 경우

---

## 3.3 의사코드

```python
Tree(X, y)
    if stopping_condition:
        return Leaf

    else:
        find_best_split로 분할 선택
        왼쪽 서브트리를 재귀적으로 생성
        오른쪽 서브트리를 재귀적으로 생성
```

---

# 4. Experimental Setup(실험 준비)

## 4.1 Dataset

- titanic dataset

## 4.2 Train / Validation split

- train - 80
- validation - 20

## 4.3 Hyperparameter Settings

하이퍼파라미터 튜닝을 통해서 최고의 값으로 설정한다.

- max_depth = 4
- min_sample_leaf = 1

---

# 5. Experimental Results(실험 결과)

## Depth와 정확도, 계산시간

| Max Depth |  Validation Accuracy   |      Duration       |
| :-------: | :--------------------: | :-----------------: |
|     1     |   0.797752808988764    | 0.03598809242248535 |
|     2     |   0.797752808988764    | 0.04854273796081543 |
|     3     |   0.8146067415730337   | 0.07001638412475586 |
|   **4**   | **0.8258426966292135** | 0.08354878425598145 |
|     5     |   0.8146067415730337   | 0.1150505542755127  |
|     6     |   0.8089887640449438   | 0.11789703369140625 |

- 계산 시간
  - max_depth가 커질수록 계산 시간은 증가한다.
  - 계산을 오래한다고 정확도가 올라가는 것은 아니며, 최고의 값을 튜닝을 통해서 찾아야 함을 알 수 있다.

## sklearn과 비교

|           Model            |      Accuracy      |
| :------------------------: | :----------------: |
| Decision Tree from Scratch | 0.8258426966292135 |
| Decision Tree from sklearn | 0.8258426966292135 |

hyperparmeter

- max_depth: 4
- min_samples_leaf: 1
- criterion: gini

Discussion

- 성능 차이
  - 성능이 같음을 알 수 있다.
- 구현 차이
  - 직접 구현의 경우에는 Gini를 기준으로 하지만 skleran의 경우에는 entrophy, gini, log_loss를 선택할 수 있게 구현되었다.

---

# 6. Bias-Variance Analysis(편향-분산 분석)

과소적합

- 원인: 최대 트리 깊이 `max_depth`가 작음.
- 분석:
  - 데이터로 부터 충분히 학습을 못해서 규칙이 단순하며, 이로 인해서 모델의 성능이 잘 나오지 않음.
  - 추론을 못하기 때문에 `편향`은 크며, `분산`은 작다.

과적합

- 원인: 최대 트리 깊이 `max_depth`가 큼.
- 분석:
  - 데이터의 이상치와 같은 값까지 학습하여 규칙은 복잡하며, 이로 인해서 훈련 데이터에 대해서는 성능이 잘 나오지만, 그 외에 데이터에 대해서는 모델의 성능이 잘 나오지 않음.
  - 데이터가 퍼지있으므로, `분산`은 크며, `편향`은 작다.

* 분산과 편향은 트레이드오프 관계를 가지고 있으므로, 분산이 크면, 편향은 작다. 반대도 성립한다.

# 7. Limitations(한계)

- 고유값 등의 데이터에 대해서 전처리나 트리에 대한 pruning을 하지 않는 경우에는 과적합되기 쉽다.
- 데이터의 인접한 값을 통해서 임계값을 정하기 때문에 데이터가 변하는 경우에는 결과가 쉽게 바뀔 수 있다(데이터에 민감함).

# 8. Future Work(향후 연구)

- Categorical Data를 Continuous Data로 변환없이 처리하는 방식
- 랜덤 포레스트 확장
- Gain 기반 알고리즘
- pruning 구현
- 최적화
- 모델을 라이브러리화

---

# 9. Conclusion

본 프로젝트에서 Decsion Tree의 이론적 배경을 정리하고, CART 알고리즘을 직접 구현하여 실험을 수행하였다.

이를 통해

- 알고리즘의 내부 동작 원리 이해

을 수행하였다.
