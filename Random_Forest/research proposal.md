# Project Proposal

## Random Forest from Scratch

---

# 1. Problem Statement(문제 정의)

본 연구는 앙상블 학습 기법 중 하나인 `Random Forest`를 `Decision Tree`를 기반으로 직접 구현하는 것을 목표로 한다. 또한 병렬 처리를 적용했을 때 학습 시간과 모델 성능에
어떤 변화가 발생하는지 분석하는 것을 목표로 한다.

---

# 2. Motivation

- 여러 모델을 결합하여 성능을 향상시키는 앙상블에 대한 이해
- 앙상블 모델 중 대표적인 `Random Forest`알고리즘의 원리와 구조에 대한 이해
- `Random Forest` 학습 과정 중 병렬처리가 가능한 이유와 그 효과에 대한 이해

---

# 3. Research Questions / Hypotesis (연구 질문 / 가설)

본 연구는 다음 가설을 검증한다:

1. 각 모델의 출력의 평균을 최종 결과로 하고 있어서 전체 분산을 낮춰서 과적합에 빠질 위험이 `Decision Tree`보다 감소할 것이다.
2. `Random Forest` 알고리즘은 트리를 독립적으로 학습하기 때문에 병렬처리를 통해서 결과를 구할 수 있을 것이다.

---

# 4. Methodology(방법론)

## 4.1 Structure(구조)

- RandomForestTree - 기존 DT에서 분할 샘플링을 추가한 버전
- RandomForest - 순차 처리 랜덤 포레스트
- ParallelRandomForest - 병렬 처리 랜덤 포레스트

## 4.2 Experimental Setup(실험 준비)

- Dataset: Titanic
  - 크기: 891

## 4.3 Evaluation Metric(평가 지표)

- 각 모델을 학습시킨 결과의 평균과 정답과의 차를 최종 성능 지표로 사용한다.
- 병렬처리를 통한 학습의 시간

# 5. Expected Outcomes(예상 결과)

- 과적합 위험이 감소하기 때문에 일반화 성능이 `Decision Tree`보다 좋을 것이다.
- 순차적으로 학습하는 `Random Forest`의 경우는 병렬처리하는 `Random Forest`보다 프로세스 당 1개 밖에 처리 못하기 때문에 느릴 것이다.

# 6. Potential Limitations(잠재적인 한계)

- 순차적 처리를 하게 되면 트리 계산이 설정한 만큼 늘어나기 때문에 계산 시간이 증가한다.

# 7. Timeline(타임라인)

| Day | Task                     |
| --- | ------------------------ |
| 1   | 이론 정리                |
|     | 알고리즘 구현            |
|     | 실험 및 튜닝             |
| 2   | 결과 분석 및 보고서 작성 |
