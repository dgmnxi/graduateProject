# Stage 1 선구현: 정규화/특징 엔지니어링

모델 아키텍처(Transformer/GCN/MLP/ResMLP) 이전 단계로, 관절값 전처리와 특징 엔지니어링만 먼저 구현했습니다.

## 구현 파일 구조

```text
ai_service/
├── services/
│   └── pose_to_beta.py              # Stage 1 서비스 진입점(전처리 연동)
└── training/
    ├── __init__.py
    ├── config.yaml                  # 데이터셋/전처리 설정 템플릿
    └── preprocessing.py             # 정규화 + 특징 엔지니어링 구현
```

## 주요 구현 내용

### 1) 관절 입력 검증/정형화
- 파일: `training/preprocessing.py`
- 입력 형태를 `(99,)` 또는 `(33, 3)`로 제한합니다.
- 형식이 다르면 명확한 예외를 발생시켜 데이터 품질 문제를 빨리 찾을 수 있게 했습니다.

### 2) 결측치 처리 (Missing Value)
- `NaN`이 있으면 축(x/y/z)별 평균으로 채웁니다.
- 축 전체가 결측이면 `0.0`으로 대체합니다.
- 처리 개수는 `metadata.missing_filled`에 기록합니다.

### 3) 정규화 (Normalization)
- 기본값: `zscore`
- 옵션: `minmax`
- 설정값은 `PreprocessConfig`와 `training/config.yaml`에 반영했습니다.

### 4) 기준점 정렬 (Pelvis Centering)
- 좌/우 골반(23, 24)의 중점을 원점으로 이동시켜 상대 좌표화합니다.
- 카메라 절대 위치 영향을 줄이기 위한 기본 처리입니다.

### 5) 이상치 완화 (Outlier Clipping)
- 정규화 후 z-score 기준 `[-3, 3]`으로 clip 합니다.
- clip 발생 횟수는 `metadata.outliers_clipped`로 기록합니다.

### 6) 특징 엔지니어링
- 거리 특징: 어깨폭, 골반폭, 팔/다리 주요 세그먼트 길이
- 각도 특징: 팔꿈치/무릎 관절각
- 대칭 특징: 좌우 팔/다리 길이 차이
- 비율 특징: 어깨-골반 비율, 팔-다리 비율, torso-hip 비율
- 최종 입력은 `flatten(normalized_joints) + feature_vector` 형태입니다.

## 서비스 연동 변경

### 파일: `services/pose_to_beta.py`
- `preprocess_joints()`가 단순 정규화 대신 `preprocess_pose()`를 호출하도록 변경했습니다.
- `predict_beta()`의 `raw_output`에 아래 디버그 정보를 포함합니다.
  - `input_shape`
  - `preprocessed_joints`
  - `feature_vector_length`
  - `metadata` (결측/이상치 처리 통계)
- 모델 추론은 아직 더미 출력(랜덤 beta) 상태를 유지했습니다.

## 데이터셋 경로 정책 반영

`training/config.yaml`에서 데이터셋 경로는 모두 빈 문자열(`""`)로 두었습니다.

- `dataset.root_dir`, `train_file`, `val_file`, `test_file`는 나중에 각자 로컬에서 채우면 됩니다.
- 상대경로 사용 전제를 문서와 주석에 명시했습니다.
- 3DPW 데이터셋 URL을 설정에 함께 기록했습니다.

## 다음 연결 포인트

1. 데이터 로더 구현 시 `training/config.yaml`에서 빈 경로를 읽어 상대경로로 해석
2. 로더 출력을 `(N, 99)`로 맞춘 후 `preprocess_pose()` 재사용
3. 이후 모델 아키텍처 단계에서 `model_input` 차원(`99 + feature_size`) 기반으로 입력층 정의
