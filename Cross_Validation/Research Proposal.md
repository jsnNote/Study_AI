# Project Proposal

## Decision Tree에서의 Cross Validation

---

# 1. Problem Statement(문제 정의)

본 연구는 `sklearn` 라이브러리를 활용하여, 'Decision Tree' 모델에서 교차 검증이 성능 평가에 미치는 영향을 분석하는 것을 목표로 한다.

---

# 2. Motivation

- 교차 검증의 원리를 이해하고자함.
- 교차 검증이 어떤 효과가 있는지 Decision Tree 모델을 통해서 분석하고자 함.

---

# 3. Research Questions / Hypotesis (연구 질문 / 가설)

본 연구는 다음 가설을 검증한다:

1.  단일 평가보다 더 안정적인 성능 평가를 제공할 것이다.
2.  트리 모델에서는 교차 검증을 사용하더라도 평균 성능 자체는 크게 변하지 않을 것이다.

---

# 4. Methodology(방법론)

## 4.1 Algorithm Design(알고리즘 디자인)

- sklearn의 DecisionTreeClassifier를 사용하여 모델을 구성
- cross_val_score를 통해서 데이터를 훈련-검증 세트 k개로 구성하고, 학습하고 정확도를 산출

## 4.2 Experimental Setup(실험 준비)

- Dataset: Titanic
- fold of #: 5 / 10 (5는 계산 비용과 안정성 균형이 좋고, 10은 더 안정적인 추정)

## 4.3 Evaluation Metric(평가 지표)

- 각 fold에서 계산된 정확도의 평균을 최종 성능 지표로 사용한다.

# 5. Expected Outcomes(예상 결과)

- 교차 검증을 사용하면 성능 지표의 변동성이 줄 것이다.
- 단일 검증하고 비교했을 때 성능은 크게 달라지지 않을 것이다.

# 6. Potential Limitations(잠재적인 한계)

- 기존의 경우에는 학습-검증을 한 번만 했다면 학습을 k번 반복해 학습해야 하므로 계산 비용이 증가한다.

# 7. Timeline(타임라인)

| Day | Task                     |
| :-- | :----------------------- |
| 1   | 이론 정리                |
|     | 알고리즘 구현            |
|     | 실험 및 튜닝             |
|     | 결과 분석 및 보고서 작성 |
