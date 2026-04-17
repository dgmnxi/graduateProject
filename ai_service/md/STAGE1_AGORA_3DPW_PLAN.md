# Stage 1 학습 및 모델 개발 계획: AGORA + 3DPW

## 1) 목표

이 단계의 목표는 사용자가 입력한 전신 사진에서 **SMPL beta(10-dim)** 를 예측하고, 그 결과를 체형 특징으로 바꿔서 패션 검색 프롬프트에 연결하는 것이다.

핵심은 이미 존재하는 완성형 모델을 그대로 붙이는 것이 아니라, **우리 데이터와 서비스 목적에 맞는 자체 beta 예측 모델**을 학습하는 것이다.

### 최종 산출물

- 전신 사진 입력 → beta(10-dim) 출력 모델 1개
- beta 차원별 MSE 분석 결과
- 모델 3종 비교 결과
- AGORA pretrain + 3DPW finetune 실험 결과
- 최종 서비스 연결용 체크포인트 1개

---

## 2) 이번 프로젝트의 학술적 의미

이 졸업프로젝트는 단순 모델 연결이 아니라 다음 질문에 답하는 실험이다.

1. 전신 사진 기반 beta 회귀가 관절-only보다 실제로 더 나은가
2. AGORA 같은 합성 데이터로 먼저 학습하면 실사 데이터에서 성능이 좋아지는가
3. MLP, Transformer, GCN 중 어떤 구조가 beta 예측에 가장 적합한가
4. beta 10개 차원 중 어떤 차원이 특히 어려운가

즉, **synthetic-to-real 전이 + 아키텍처 비교 + 차원별 오차 분석**이 이 프로젝트의 연구 포인트다.

---

## 3) 서비스 시나리오에 맞는 전체 흐름

1. 사용자가 특정 자세로 서 있는 **전신 사진**을 입력한다.
2. 모델이 그 사진에서 사람의 체형을 읽고 **SMPL beta(10-dim)** 를 예측한다.
3. beta를 `마름 / 보통 / 비만` 같은 체형 특성으로 변환한다.
4. 체형 특성과 사용자의 태그를 합쳐 프롬프트를 만든다.
5. 이 프롬프트를 Marqo에 넣어 의류를 검색한다.

따라서 Stage 1의 핵심은 **image-to-beta** 모델을 안정적으로 만드는 것이다.

---

## 4) 데이터셋 역할

### AGORA 720p

- 역할: **pretrain용 메인 데이터셋**
- 이유: 체형 다양성이 넓고 beta 라벨 품질이 좋다
- 기대 효과: 모델이 체형의 폭을 먼저 학습한다

### 3DPW

- 역할: **finetune / validation / final test**
- 이유: 실사 환경에 가깝고 서비스 입력과 더 비슷하다
- 기대 효과: AGORA에서 배운 체형 표현을 실제 서비스 도메인으로 옮긴다

### 권장 사용법

- pretrain: AGORA
- finetune: 3DPW train
- validation: 3DPW val
- test: 3DPW test

---

## 5) 입력 설계

이번 계획에서는 얼굴 면적 같은 파생 feature는 주력으로 쓰지 않는다. 이전 실험에서 설명력이 낮았기 때문이다.

### 주 입력

- 전신 이미지
- 사람 bbox 또는 crop 정보
- 필요하면 pose heatmap 또는 관절 보조 정보

### 제외 또는 보조로만 둘 입력

- 얼굴 면적
- 얼굴 bbox 비율만으로 만든 단순 파생값
- 관절만으로 만든 약한 feature들

핵심은 **체형 정보가 더 잘 드러나는 입력 표현**으로 바꾸는 것이다.

---

## 6) 비교할 모델 3개

이번 Stage 1에서는 아래 3개만 먼저 비교한다.

### 1. MLP

- 가장 단순한 baseline
- 이미지 backbone에서 뽑은 feature를 회귀하는 방식
- 목적: 빠른 기준선 확보

### 2. Transformer

- 전신 이미지의 전역 관계를 학습하는 구조
- 목적: 신체 부위 간 관계를 더 잘 잡는지 확인

### 3. GCN

- 관절 구조나 skeleton 구조를 활용하는 모델
- 목적: 인체 구조 정보를 명시적으로 쓰면 성능이 좋아지는지 확인

### 모델 선택 기준

- 전체 MSE
- beta별 MSE
- 학습 안정성
- 추론 속도
- 3DPW test 성능

---

## 7) 학습 전략

### Phase 1. AGORA pretrain

목적은 체형의 다양성을 먼저 배우게 하는 것이다.

권장 방식:

- 입력: 전신 이미지
- 출력: beta 10-dim
- 손실: MSE 또는 L1 + MSE 혼합
- 학습: 10~20 epoch 정도의 초기 pretrain

### Phase 2. 3DPW finetune

목적은 실사 환경으로 옮기는 것이다.

권장 방식:

- AGORA checkpoint 로드
- learning rate를 낮춘다
- backbone 일부 freeze 후 head부터 학습한다
- validation 기준으로 early stopping을 쓴다

### Phase 3. 최종 비교

- MLP / Transformer / GCN을 같은 조건으로 비교
- AGORA pretrain 유무도 같이 비교
- 최종적으로 가장 안정적인 모델 1개를 고른다

---

## 8) 실험 순서

### Step 1. 데이터 준비

- [ ] AGORA와 3DPW를 같은 포맷으로 맞춘다
- [ ] train / val / test split을 만든다
- [ ] 이미지 crop 규칙을 통일한다
- [ ] beta 10-dim 라벨을 확인한다

### Step 2. baseline 구축

- [ ] MLP 학습
- [ ] Transformer 학습
- [ ] GCN 학습
- [ ] 각 모델의 test MSE 확인

### Step 3. AGORA pretrain

- [ ] AGORA로 각 모델을 먼저 학습
- [ ] checkpoint 저장
- [ ] 전신 이미지 기반 표현이 제대로 학습되는지 확인

### Step 4. 3DPW finetune

- [ ] AGORA checkpoint를 3DPW에 맞게 미세조정
- [ ] validation에서 모델 선택
- [ ] beta 차원별 오차 분석

### Step 5. 최종 선택

- [ ] 가장 성능이 좋은 모델 1개 선택
- [ ] 서비스 코드에 연결할 checkpoint 저장
- [ ] 프롬프트 생성 단계와 연결 준비

---

## 9) 평가 지표

### 필수 지표

- MSE
- RMSE
- R²
- beta 0~9 각각의 MSE

### 추가로 보면 좋은 지표

- beta별 MAE
- 추론 시간
- 모델 크기
- pretrain 전후 성능 차이

---

## 10) 성공 기준

아래 조건을 만족하면 Stage 1이 의미 있는 결과라고 볼 수 있다.

- 전체 beta MSE가 기존 관절-only보다 확실히 낮아진다
- beta 차원별로도 성능이 지나치게 무너지지 않는다
- AGORA pretrain이 3DPW only보다 도움이 된다
- 세 모델 중 1개가 가장 안정적으로 선택된다

목표는 beta 각 차원의 MSE를 가능한 한 **0~0.25 범위**에 가깝게 낮추는 것이다.

---

## 11) 체크리스트

### 데이터

- [ ] AGORA 다운로드 가능 여부 확인
- [ ] 3DPW train / val / test 위치 확인
- [ ] 이미지 전처리 규칙 정리
- [ ] beta 라벨 정합 확인

### 모델

- [ ] MLP 구조 결정
- [ ] Transformer 구조 결정
- [ ] GCN 구조 결정
- [ ] 동일한 출력 차원(10-dim) 유지

### 학습

- [ ] AGORA pretrain 실행
- [ ] 3DPW finetune 실행
- [ ] validation 기반 checkpoint 저장
- [ ] seed 고정

### 평가

- [ ] 전체 MSE 비교
- [ ] beta별 MSE 비교
- [ ] 최종 모델 1개 선택
- [ ] 서비스 연결 준비

---

## 12) 최종 결론

이번 Stage 1은 "기존 모델을 가져다 쓰는 작업"이 아니다.

정확히는 다음이다.

- 전신 사진에서 beta를 예측하는 **우리만의 회귀 모델**을 만든다
- AGORA로 체형 다양성을 먼저 학습한다
- 3DPW로 실사 도메인에 맞게 조정한다
- MLP / Transformer / GCN 중 가장 나은 구조를 선택한다
- 그 결과를 Marqo 검색 프롬프트 생성의 앞단에 연결한다

이렇게 해야 졸업프로젝트로서 학술적 의미가 생긴다.