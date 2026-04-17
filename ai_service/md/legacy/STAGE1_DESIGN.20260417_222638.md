# Stage 1: MediaPipe 관절값 → SMPL Beta 값 변환 설계 계획서

## 📋 개요

MediaPipe의 33개 관절 좌표 데이터로부터 SMPL 모델의 10개 beta 값(체형 파라미터)을 예측하는 Stage 1의 구현 계획입니다.

**목표**: 관절값 → Beta값 변환 모델을 4가지 아키텍처로 구축하고 성능을 비교 분석

---

## 1. 시스템 아키텍처

### 1.1 입출력 명세

```
INPUT:
- joints: List[float] (33개 관절 × 3차원좌표 = 99개 값)
  → 각 관절: (x, y, z) 좌표

PREPROCESSING:
1. 결실값 처리 (Missing values)
2. 정규화 (Normalization)
   - Z-score normalization 또는 Min-Max normalization
3. 이상치 제거 (Outlier removal)
4. 관절간 거리/각도 특징 추출 (Optional)

OUTPUT:
- beta_values: List[float] (10개)
  - beta[0]: 체형 크기 (size)
  - beta[1-2]: 길쭉함/뚱뚱함 (shape)
  - beta[3-9]: 세부 체형 특징 (detail shape features)
- confidence: float (0~1)
- metadata: Dict (처리 정보)
```

### 1.2 파이프라인 구조

```
MediaPipe 관절값 (33×3=99)
        ↓
    전처리 (Preprocessing)
   ├─ Z-score 정규화
   ├─ 관절 신뢰도 필터링
   └─ 특징 엔지니어링 (optional)
        ↓
   입력 특징값 (Variable)
     ├─ 모델1: Transformer → beta (10)
     ├─ 모델2: GCN → beta (10)
     ├─ 모델3: MLP → beta (10)
     └─ 모델4: ResMLP → beta (10)
        ↓
    후처리 (Post-processing)
   ├─ Beta값 범위 제약 (-3~3)
   └─ 신뢰도 계산
        ↓
  최종 출력: {beta, confidence, metadata}
```

### 1.3 1차 실험 결론 및 대체 전략

```
실험 관찰:
- MediaPipe 관절값(99) 단독 입력으로 4개 모델(Transformer/GCN/MLP/ResMLP) 학습 시
   MSE가 0.6 이상으로 STAR/SMPL 체형 파라미터(beta) 활용에 부적합.

해석:
- 2D/약한 3D 관절 좌표만으로는 체형(둘레, 부피, 지방 분포) 정보를 충분히 복원하기 어려움.
- 특히 카메라 거리, 원근, 포즈 변화가 체형 신호를 덮어쓰는 경향이 큼.
```

대체 전략(검증 우선순위):

1. 얼굴 면적 기반 feature 추가 검증
    - face_area: 얼굴 landmark bbox 면적
    - face_to_torso_area_ratio: 얼굴/상체 bbox 면적 비율
    - face_to_shoulder_ratio: 얼굴 폭/어깨 폭 비율
2. 상관 분석으로 유의성 확인 후 모델 입력 채택 여부 결정
    - Pearson, Spearman, 단일 feature 선형회귀 R² 계산
3. 유의성이 낮으면 이미지 기반 body-shape encoder로 전환
    - 전신 segmentation/silhouette, depth 추정, multi-view 정보 결합 검토

검증 스크립트:

```bash
python training/analyze_face_beta_correlation.py \
   --data data/processed/train_data.npz data/processed/validation_data.npz data/processed/test_data.npz \
   --output ai_service/outputs/visual/face_beta_correlation_report.md
```

판단 기준(권장):

- |corr| < 0.2: 실질적 예측 신호가 약함 (feature 채택 비권장)
- 0.2 <= |corr| < 0.35: 약한 신호 (보조 feature로만 사용)
- |corr| >= 0.35: 중간 이상 신호 (모델 입력 채택 후보)
- 단일 feature R² < 0.1이면 STAR beta 직접 추정 feature로는 한계 가능성 높음

---

## 2. 데이터 전처리 상세 설계

### 2.1 정규화 (Normalization)

```python
# 구현 전략
def preprocess_joints(joints: np.ndarray) -> np.ndarray:
    """
    입력: (99,) or (batch, 99)
    출력: (99,) or (batch, 99) - 정규화된 값
    
    단계:
    1. Z-score normalization
       normalized = (x - mean) / std
    
    2. 관절별 정규화 (선택사항)
       - 좌표계 기준점 설정 (e.g., 골반 관절)
       - 상대 좌표로 변환
    
    3. 이상치 제거
       - z-score > 3σ 값 제거 또는 clipping
    """
    pass
```

### 2.2 특징 엔지니어링 (Feature Engineering)

```python
def extract_features(joints: np.ndarray) -> np.ndarray:
    """
    선택사항: 관절값 이외의 특징 추가
    
    추출 가능 특징:
    1. 관절간 거리
       - 각 관절 쌍의 거리 계산
       - 예: 어깨-팔꿈치 거리, 팔꿈치-손목 거리 등
    
    2. 관절간 각도
       - 3개 관절로 이루어진 각도
       - 예: 어깨-팔꿈치-손목 각도
    
    3. 신체 비율
       - 상반신 길이 / 하반신 길이 등
       - 팔 길이 / 다리 길이 등
    
    4. 대칭성 특징
       - 좌측과 우측의 관절 차이
    
    최종 입력: 99 + α 개의 특징값
    """
    pass
```

---

## 3. 모델 아키텍처 설계

### 3.1 모델 1: Transformer 기반 (TransformerPoseToBeta)

#### 구조
```
입력: (batch, 99)
   ↓
임베딩층: (batch, 99) → (batch, 99, embed_dim=128)
   ↓
Positional Encoding 추가
   ↓
Transformer Encoder (L=4層)
   ├─ Multi-head Self-Attention (heads=8)
   ├─ Feed Forward 네트워크
   └─ LayerNorm + Residual Connection
   ↓
전역 풀링 또는 [CLS] 토큰: (batch, 128)
   ↓
Output MLP (128 → 64 → 32 → 10)
   ↓
출력: (batch, 10) - beta 값
```

#### 특징
- **장점**:
  - 전역 관절 관계를 모델링
  - Self-attention으로 중요한 관절 쌍 학습
  - 순서불변성(permutation invariant)
  
- **단점**:
  - 계산 비용 높음 (O(n²) attention)
  - 학습 데이터 많이 필요
  - 관절의 그래프 구조를 명시적으로 사용 안함

#### 초매개변수
```yaml
embed_dim: 128
num_heads: 8
num_layers: 4
feedforward_dim: 256
dropout: 0.1
activation: 'gelu'
```

---

### 3.2 모델 2: Graph Convolutional Network 기반 (GCNPoseToBeta)

#### 구조
```
입력: (batch, 99)
   ↓
노드 특징 변환: (batch, 99) → (batch, 33, 3) → (batch, 33, feat_dim=64)
   ↓
그래프 정의 (MediaPipe 골격 그래프)
   ├─ 33개 노드 (관절)
   └─ E개 엣지 (뼈대 연결)
   ↓
GCN 레이어 (L=3層)
   ├─ Graph Conv: feat_dim → hidden_dim
   ├─ BatchNorm + ReLU
   └─ Residual Connection (선택)
   ↓
노드 특징 집계: (batch, 33, hidden_dim) → (batch, hidden_dim)
   ├─ 평균 풀링 또는 어텐션 풀링
   ↓
Output MLP (hidden_dim → 64 → 32 → 10)
   ↓
출력: (batch, 10) - beta 값
```

#### 특징
- **장점**:
  - 인체의 골격 그래프 구조를 활용
  - 생물학적으로 타당함
  - 계산 비용 낮음 (O(E))
  - 명시적 관절 연결 관계 모델링
  
- **단점**:
  - 순차 정보 고려 안함
  - 그래프 정의의 유연성 제한

#### MediaPipe 골격 그래프
```
33개 관절:
- 두부 (head): 11개 관절
- 상반신 (torso): 8개 관절
- 좌팔 (left_arm): 5개 관절
- 우팔 (right_arm): 5개 관절
- 좌다리 (left_leg): 2개 관절
- 우다리 (right_leg): 2개 관절

주요 연결 (엣지):
- 귀 ↔ 어깨 ↔ 팔꿈치 ↔ 손목 등 (약 30~40개 엣지)
```

#### 초매개변수
```yaml
feat_dim: 64
hidden_dim: 128
num_layers: 3
dropout: 0.1
activation: 'relu'
graph_type: 'mediapipe'  # bone structure
aggregation: 'mean'  # 'mean', 'attention', 'max'
```

---

### 3.3 모델 3: Multi-Layer Perceptron (MLPPoseToBeta)

#### 구조
```
입력: (batch, 99)
   ↓
Hidden Layer 1: 99 → 256
   ├─ Linear
   ├─ BatchNorm
   ├─ ReLU
   └─ Dropout(0.2)
   ↓
Hidden Layer 2: 256 → 128
   ├─ Linear
   ├─ BatchNorm
   ├─ ReLU
   └─ Dropout(0.2)
   ↓
Hidden Layer 3: 128 → 64
   ├─ Linear
   ├─ BatchNorm
   ├─ ReLU
   └─ Dropout(0.1)
   ↓
Output Layer: 64 → 10
   ↓
출력: (batch, 10) - beta 값
```

#### 특징
- **장점**:
  - 구현 간단, 학습 빠름
  - 계산 비용 매우 낮음 (O(n))
  - 해석 용이
  - 작은 데이터셋에서도 학습 가능
  
- **단점**:
  - 관절 관계 구조 무시
  - 공간 정보 활용 못함
  - 표현력 제한

#### 초매개변수
```yaml
hidden_sizes: [256, 128, 64]
dropout_rates: [0.2, 0.2, 0.1]
activation: 'relu'
batch_norm: True
```

---

### 3.4 모델 4: Residual MLP 기반 (ResMLPPoseToBeta)

#### 구조
```
입력: (batch, 99)
   ↓
Residual Block 1: 99 → 256
   ├─ Linear(99→256) + BatchNorm + ReLU
   ├─ Linear(256→256) + BatchNorm + ReLU
   ├─ Residual Connection (입력 + 출력)
   └─ Dropout(0.2)
   ↓
Residual Block 2: 256 → 128
   ├─ Linear(256→128) + BatchNorm + ReLU
   ├─ Linear(128→128) + BatchNorm + ReLU
   ├─ Residual Connection (입력 + 출력)
   └─ Dropout(0.2)
   ↓
Residual Block 3: 128 → 64
   ├─ Linear(128→64) + BatchNorm + ReLU
   ├─ Linear(64→64) + BatchNorm + ReLU
   ├─ Residual Connection (입력 + 출력)
   └─ Dropout(0.1)
   ↓
Output Layer: 64 → 10
   ↓
출력: (batch, 10) - beta 값
```

#### 특징
- **장점**:
  - Residual connection으로 깊은 네트워크 학습 가능
  - Gradient vanishing 문제 완화
  - MLP보다 더 복잡한 패턴 학습 가능
  - 계산 비용은 MLP와 유사 (O(n))
  
- **단점**:
  - MLP보다 파라미터 수 증가
  - 구현 복잡도 증가
  - 과적합 가능성 존재

#### 초매개변수
```yaml
hidden_sizes: [256, 128, 64]
dropout_rates: [0.2, 0.2, 0.1]
activation: 'relu'
batch_norm: True
num_blocks_per_layer: 2  # 각 residual block당 서브레이어 수
use_projection: False    # 차원 불일치 시 projection 사용 여부
```

---

## 4. 모델 구현 세부 계획

### 4.1 파일 구조

```
ai_service/
├── models/
│   ├── __init__.py
│   ├── base_model.py          # 추상 베이스 클래스
│   ├── transformer_model.py   # TransformerPoseToBeta
│   ├── gcn_model.py           # GCNPoseToBeta
│   ├── mlp_model.py           # MLPPoseToBeta
│   └── resmlp_model.py        # ResMLPPoseToBeta
├── training/
│   ├── __init__.py
│   ├── config.yaml            # 학습 설정
│   ├── train.py               # 학습 스크립트
│   ├── compare.py             # 모델 성능 비교 스크립트
│   └── utils.py               # 학습 유틸리티
├── STAGE1_DESIGN.md           # 본 문서
└── ...
```

### 4.2 베이스 모델 클래스 (base_model.py)

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePoseToNetaModel(ABC, nn.Module):
    """
    모든 모델의 베이스 클래스
    
    속성:
    - input_size: 99 (관절 좌표)
    - output_size: 10 (beta 값)
    - model_name: str
    
    메서드:
    - forward(joints: Tensor) -> Tensor
    - preprocess_input(joints: np.ndarray) -> Tensor
    - postprocess_output(output: Tensor) -> np.ndarray
    """
    
    def __init__(self, input_size=99, output_size=10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
    
    @abstractmethod
    def forward(self, x):
        """모델 순전파"""
        pass
    
    def preprocess_input(self, joints):
        """NumPy 배열 → Tensor 변환"""
        pass
    
    def postprocess_output(self, output):
        """Tensor → NumPy 배열 변환 + Beta 범위 제약"""
        pass
    
    def get_model_info(self):
        """모델 정보 반환"""
        pass
```

### 4.3 각 모델 구현 계획

#### TransformerPoseToBA (transformer_model.py)
```python
class TransformerPoseToBA(BasePoseToNetaModel):
    """
    구현 요소:
    1. 임베딩층 (ProjectionLayer)
    2. Positional Encoding (Sine/Cosine 또는 Learnable)
    3. TransformerEncoder 블록
       - MultiHeadAttention
       - FeedForward
       - LayerNorm
       - Residual Connection
    4. 글로벌 풀링
    5. Output MLP
    """
    pass
```

#### GCNPoseToBA (gcn_model.py)
```python
class GCNPoseToBA(BasePoseToNetaModel):
    """
    구현 요소:
    1. 노드 임베딩층
    2. MediaPipe 그래프 정의 (adjacency matrix)
    3. GraphConvolution 레이어
       - Graph Conv 연산
       - 활성화 함수
       - Batch Normalization
    4. 노드 특징 집계 (Pooling)
    5. Output MLP
    """
    pass
```

#### MLPPoseToBA (mlp_model.py)
```python
class MLPPoseToBA(BasePoseToNetaModel):
    """
    구현 요소:
    1. 선형층 시퀀스
    2. Batch Normalization
    3. ReLU/GELU 활성화
    4. Dropout
    5. Output 층
    """
    pass
```

#### ResMLPPoseToBA (resmlp_model.py)
```python
class ResMLPPoseToBA(BasePoseToNetaModel):
    """
    구현 요소:
    1. Residual Block 클래스
       - 두 개의 선형층 + BatchNorm + ReLU
       - Residual Connection (입력 + 출력)
       - Dropout
    2. 여러 Residual Block 스택
    3. Output MLP
    4. 차원 불일치 처리 (projection)
    """
    pass
```

---

## 5. 학습 설계

### 5.1 학습 설정 (training/config.yaml)

```yaml
# 데이터
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  batch_size: 32
  num_workers: 4
  normalization: 'zscore'  # 'zscore' or 'minmax'

# 학습
training:
  epochs: 100
  learning_rate: 0.001
  optimizer: 'adam'  # 'adam', 'sgd'
  lr_scheduler: 'cosine'  # 'cosine', 'step', 'exponential'
  weight_decay: 0.0001
  gradient_clip: 1.0

# 손실 함수
loss:
  type: 'mse'  # 'mse', 'huber', 'smooth_l1'
  reduction: 'mean'

# 모델 관련
models:
  transformer:
    embed_dim: 128
    num_heads: 8
    num_layers: 4
    feedforward_dim: 256
    dropout: 0.1
  
  gcn:
    feat_dim: 64
    hidden_dim: 128
    num_layers: 3
    dropout: 0.1
    aggregation: 'mean'
  
  mlp:
    hidden_sizes: [256, 128, 64]
    dropout_rates: [0.2, 0.2, 0.1]
  
  resmlp:
    hidden_sizes: [256, 128, 64]
    dropout_rates: [0.2, 0.2, 0.1]
    num_blocks_per_layer: 2
    use_projection: False

# 체크포인트
checkpoint:
  save_interval: 10
  save_dir: './checkpoints'
  keep_best: True

# 로깅
logging:
  log_interval: 10
  log_dir: './logs'
  use_tensorboard: True
```

### 5.2 학습 스크립트 (training/train.py)

```python
"""
실행 방식:
python training/train.py --model transformer --config training/config.yaml
python training/train.py --model gcn --config training/config.yaml
python training/train.py --model mlp --config training/config.yaml
python training/train.py --model resmlp --config training/config.yaml

주요 기능:
1. 데이터 로드 및 전처리
2. 모델 초기화
3. 학습 루프
   - Forward pass
   - Loss 계산
   - Backward pass
   - Optimizer step
4. 검증 및 평가
5. 체크포인트 저장
6. 로깅 (TensorBoard)
"""
```

### 5.3 손실 함수 설계

```python
def loss_fn(output, target):
    """
    MSE Loss (기본):
    L = 1/N * Σ(output_i - target_i)²
    
    선택 옵션:
    1. L1 Loss (절댓값): 이상치에 덜 민감
    2. Smooth L1 Loss: MSE와 L1의 조합
    3. Huber Loss: 강건성
    4. 카스텀 Loss: Beta값 범위 제약
    """
    pass
```

---

## 6. 모델 성능 비교 설계

### 6.1 평가 지표

```python
# 회귀 문제이므로 다음 지표 사용:

1. MAE (Mean Absolute Error)
   - 평균 절댓값 오차
   - MAE = 1/N * Σ|y_true - y_pred|

2. RMSE (Root Mean Squared Error)
   - 제곱근 평균 제곱 오차
   - RMSE = sqrt(1/N * Σ(y_true - y_pred)²)

3. R² Score (결정 계수)
   - 모델이 설명하는 분산의 비율
   - R² = 1 - (SS_res / SS_tot)
   - 범위: 0~1 (1에 가까울수록 좋음)

4. 각 Beta별 RMSE
   - Beta[0], Beta[1], ..., Beta[9] 각각의 RMSE
   - 어느 beta 값을 잘/못 예측하는지 파악

5. 추론 시간 (Inference Time)
   - 배치당 평균 추론 시간 (ms)
   - 모델 크기 (파라미터 수)

6. 시각화
   - 실제값 vs 예측값 산점도
   - 잔차 (residual) 그래프
   - Beta별 오차 분포 히스토그램
```

### 6.2 비교 스크립트 (training/compare.py)

```python
"""
실행:
python training/compare.py \
  --models transformer gcn mlp resmlp \
  --test_data path/to/test_data.pt \
  --output results/comparison.json \
  --visualize

주요 기능:
1. 각 모델별 평가 지표 계산
2. 비교 표 생성
3. 성능 차트 생성
4. 추론 시간 측정
5. 메모리 사용량 측정
6. 결과 저장 (JSON, CSV, 이미지)

출력:
- comparison/metrics.json
  {
    "transformer": {"MAE": ..., "RMSE": ..., "R2": ..., ...},
    "gcn": {...},
    "mlp": {...},
    "resmlp": {...}
  }

- comparison/comparison.png
  ├─ 모델별 RMSE 비교 바 차트
  ├─ 모델별 R² 스코어 비교
  ├─ 추론 시간 비교
  └─ 모델 크기 비교

- comparison/detailed_results.html
  상세 평가 결과 웹 리포트
"""
```

### 6.3 비교 항목

| 항목 | 설명 | 측정 방법 |
|------|------|----------|
| **정확도** | MAE, RMSE, R² | 테스트셋에서 계산 |
| **속도** | 추론 시간 | 1000개 샘플로 측정 |
| **메모리** | 모델 크기 | 파라미터 수 계산 |
| **안정성** | 검증셋 성능 | Early stopping 추적 |
| **수렴성** | 학습 곡선 | 손실 함수 추이 |
| **일반화** | 오버피팅 정도 | (Train RMSE - Val RMSE) |
| **해석성** | 어떤 관절이 중요한지 | Attention weight 또는 Gradient-based 방법 |

---

## 7. 데이터셋 요구사항

### 7.1 데이터 포맷

```python
"""
학습 데이터 포맷:

1. 입력데이터 (X)
   - Shape: (N, 99) 또는 (N, 33, 3)
   - N: 샘플 수
   - 99: 33개 관절 × 3 좌표값 (x, y, z)
   - 범위: 임의 (전처리에서 정규화)

2. 타겟데이터 (y)
   - Shape: (N, 10)
   - 10: SMPL beta 값
   - 범위: [-3, 3] (SMPL 표준)

3. 메타데이터 (optional)
   - 신뢰도 점수 (confidence)
   - 나이, 성별 등 정보
   - 데이터 소스
"""
```

### 7.2 데이터 요구량

```
권장사항:
- 최소: 1,000개 샘플
- 보통: 10,000개 샘플
- 최적: 100,000개 이상

분할:
- 학습셋: 70% (7,000 ~ 70,000개)
- 검증셋: 15% (1,500 ~ 15,000개)
- 테스트셋: 15% (1,500 ~ 15,000개)
```

---

## 8. 구현 순서 및 일정

### Phase 1: 기초 설정 (1주)
- [ ] 데이터 로더 구현
- [ ] 전처리 모듈 구현
- [ ] 베이스 모델 클래스 정의

### Phase 2: 모델 구현 (2주)
- [ ] MLP 모델 구현 및 테스트
- [ ] ResMLP 모델 구현 및 테스트
- [ ] Transformer 모델 구현 및 테스트
- [ ] GCN 모델 구현 및 테스트

### Phase 3: 학습 파이프라인 (1주)
- [ ] 학습 스크립트 구현
- [ ] 손실 함수 및 옵티마이저 설정
- [ ] 검증 및 체크포인트 로직

### Phase 4: 성능 평가 (1주)
- [ ] 평가 지표 계산 구현
- [ ] 비교 스크립트 구현
- [ ] 시각화 및 리포팅

### Phase 5: 최적화 및 튜닝 (1주)
- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 경량화 (optional)
- [ ] 최종 성능 평가

---

## 9. 예상 성능 및 기대효과

### 9.1 성능 예상

```
모델별 예상 R² 스코어 (가정):
- MLP: 0.75 ~ 0.82 (빠르고 기본적)
- ResMLP: 0.80 ~ 0.87 (Residual로 향상)
- GCN: 0.82 ~ 0.89 (구조 활용)
- Transformer: 0.85 ~ 0.92 (최고 성능 기대)

추론 시간 (1개 샘플 기준):
- MLP: < 1ms
- ResMLP: 1~2ms
- GCN: 1~3ms
- Transformer: 3~10ms
```

### 9.2 기대 효과

1. **모델 비교를 통한 인사이트**
   - 어떤 구조가 이 문제에 가장 적합한가?
   - 런타임과 정확도의 트레이드오프 분석

2. **향후 개선 방향 제시**
   - 앙상블 모델 고려
   - 전이학습 가능성 검토

3. **재사용 가능한 기반 코드**
   - Stage 2-4의 베이스
   - 다른 회귀 문제에도 적용 가능

---

## 10. 주의사항 및 고려사항

### 10.1 데이터 관련
- [ ] 데이터셋 불균형 처리 (가중치 조정)
- [ ] 결실값 처리 방법 결정
- [ ] 데이터 증강 (augmentation) 고려

### 10.2 모델 관련
- [ ] GPU 메모리 제약 (배치 크기 조정)
- [ ] 모델 저장/로드 형식 (ONNX? TorchScript?)
- [ ] 프로덕션 배포 고려 (모델 경량화)

### 10.3 학습 관련
- [ ] Early stopping 조건 설정
- [ ] 불안정한 학습 (gradient exploding/vanishing) 대응
- [ ] 하이퍼파라미터 초기화 전략

### 10.4 평가 관련
- [ ] 교차 검증 (k-fold) 고려
- [ ] 통계적 유의성 검정 (t-test 등)
- [ ] 이상치 분석 (outlier analysis)

---

## 11. 참고 자료 및 기술 스택

### 기술 스택
```
- PyTorch: 딥러닝 프레임워크
- NumPy: 수치 연산
- Pandas: 데이터 처리
- Matplotlib/Seaborn: 시각화
- TensorBoard: 학습 모니터링
- ONNX: 모델 표준화 (Optional)
```

### 관련 논문 및 자료
- SMPL: A Skinned Multi-Person Linear Model (Loper et al., 2015)
- Attention Is All You Need (Vaswani et al., 2017)
- Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, 2016)
- MediaPipe Pose: On-device Real-time Body Pose Estimation

---

## 12. 향후 확장 계획

### 단계별 발전
1. **단일 모델 최적화**: 가장 성능 좋은 모델 집중 개선
2. **앙상블 모델**: 여러 모델의 예측값 결합
3. **멀티태스크 학습**: Beta값 외 추가 정보(신뢰도, 활동성 등) 동시 학습
4. **온라인 학습**: 사용자 피드백으로 실시간 모델 업데이트

---

## 요약

이 설계 계획서는 Stage 1 구현의 전체적인 틀을 제시합니다:
- **4가지 모델**: MLP (기본), ResMLP (Residual 향상), GCN (구조활용), Transformer (최고 기대)
- **체계적 비교**: 정량적 지표로 객관적 평가
- **확장 가능성**: 베이스 클래스 기반 모듈식 설계
- **실용성**: 프로덕션 배포를 고려한 구조

각 모델의 강점을 활용하여 가장 적합한 솔루션을 찾을 수 있을 것으로 예상됩니다.
