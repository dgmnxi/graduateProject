# 데이터셋 정리 (Stage 1: AGORA + 3DPW)

이 문서는 Stage 1 계획에 맞춰 AGORA와 3DPW를 함께 관리하는 기준 문서다.

핵심 전략:

- pretrain: AGORA
- finetune: 3DPW train
- validation: 3DPW val
- test: 3DPW test

---

## 1. 데이터셋 역할

### AGORA

- 용도: pretrain용 메인 데이터셋
- 목적: 체형 다양성 학습
- 기대 효과: beta 회귀의 초기 표현력 확보

### 3DPW

- 용도: finetune, validation, final test
- 목적: 실사 도메인 성능 확보
- 기대 효과: 서비스 입력 환경 일반화

---

## 2. 데이터 저장 구조

### 2-1. Raw (원본 보관)

```text
data/
  raw/
    agora/
      images/
      annotations/
      splits/
    3dpw/
      images/
      annotations/
      splits/
```

원칙:

- raw는 원본 보관용이며 직접 수정하지 않는다.
- 전처리 결과는 반드시 processed에 저장한다.

### 2-2. Processed (학습 입력)

```text
data/
  processed/
    agora/
      train_data.npz
      val_data.npz
    3dpw/
      train_data.npz
      validation_data.npz
      test_data.npz
```

참고:

- 현재 루트의 data/processed/train_data.npz, validation_data.npz, test_data.npz는 기존 3DPW 단일 구성 산출물이다.
- Stage 1 실험에서는 데이터셋을 분리한 하위 구조를 권장한다.

---

## 3. NPZ 포맷 정의

각 NPZ 파일에는 아래 2개 배열이 저장된다.

### 3-1. joints 배열

- 의미: MediaPipe 기반 포즈 관절 좌표
- shape: (N_samples, 99)
- 구성: 33개 관절 x 3좌표(x, y, z)
- dtype: float32

### 3-2. betas 배열

- 의미: SMPL 신체 형태 파라미터
- shape: (N_samples, 10)
- 구성: beta 10개 차원
- dtype: float32

---

## 4. 현재 보유 통계 (기존 3DPW 단일 processed 기준)

아래 통계는 기존 파일(data/processed/*.npz) 기준이며,
Stage 1 분리 구조로 재전처리 후 갱신이 필요하다.

### Train Dataset (train_data.npz)

| 항목 | 값 |
|------|-----|
| 파일 크기 | 3.98 MB |
| 샘플 수 | 9,570 |
| joints shape | (9570, 99) |
| betas shape | (9570, 10) |
| joints 통계 | min=-1.6517, max=1.3132, mean=0.2685 |
| betas 통계 | min=-0.7518, max=1.4049, mean=-0.0031 |

### Validation Dataset (validation_data.npz)

| 항목 | 값 |
|------|-----|
| 파일 크기 | 1.31 MB |
| 샘플 수 | 3,141 |
| joints shape | (3141, 99) |
| betas shape | (3141, 10) |
| joints 통계 | min=-1.6127, max=1.1310, mean=0.2623 |
| betas 통계 | min=-0.7518, max=1.4049, mean=-0.0009 |

### Test Dataset (test_data.npz)

| 항목 | 값 |
|------|-----|
| 파일 크기 | 0.18 MB |
| 샘플 수 | 440 |
| joints shape | (440, 99) |
| betas shape | (440, 10) |
| joints 통계 | min=-1.0831, max=1.0067, mean=0.3630 |
| betas 통계 | min=-0.0674, max=0.0456, mean=-0.0111 |

---

## 5. Stage 1 전처리/검증 체크리스트

### 데이터 준비

- [ ] AGORA raw 경로(images, annotations, splits) 정합 확인
- [ ] 3DPW raw 경로(images, annotations, splits) 정합 확인
- [ ] AGORA/3DPW 라벨(beta 10-dim) 스키마 일치 확인

### 전처리

- [ ] AGORA processed 생성(train/val)
- [ ] 3DPW processed 생성(train/validation/test)
- [ ] 각 npz의 joints, betas shape 검증

### 학습 연계

- [ ] AGORA pretrain 데이터 로딩 확인
- [ ] 3DPW finetune/validation/test 로딩 확인
- [ ] 모델별 MSE와 beta별 MSE 로그 저장 확인

---

## 6. 데이터 출처

- AGORA: synthetic 인체 데이터셋 (pretrain용)
- 3DPW: 실사 인체 데이터셋 (finetune/평가용)

처리 과정(공통):

1. 원본 이미지에서 pose/신체 표현 추출
2. 입력 feature를 모델 입력 포맷으로 정렬
3. SMPL beta(10-dim) 라벨과 매칭
4. NPZ 형식으로 저장

---

## 7. 문서 연계

- Stage 1 목표/평가 기준: ai_service/md/STAGE1_AGORA_3DPW_PLAN.md
- 데이터 구조 설정 가이드: ai_service/md/DATASET_STRUCTURE_SETUP_GUIDE.md
