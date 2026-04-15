# 시각화용 스니펫

이 문서는 `train.py` 실행 후 생성되는 결과 JSON을 Notebook에서 직접 시각화할 때 사용하는 코드 모음이다.

전제:
- 학습 결과는 `ai_service/outputs/checkpoints/YYYYMMDD[_HHMMSS]_MODEL/` 아래에 저장된다.
- 시각화는 자동 실행하지 않고, Notebook 또는 별도 스크립트에서 수동으로 실행한다.
- 프로젝트 `requirements.txt`에는 시각화 라이브러리를 추가하지 않는다. 필요 시 Notebook 환경에서 개별 설치한다.

---

## 1) 공통 준비 코드

```python
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
```

```python
project_root = Path(r"a:\graduate\graduateProject")
outputs_root = project_root / "ai_service" / "outputs"
checkpoints_root = outputs_root / "checkpoints"
visual_root = outputs_root / "visual"
```

---

## 2) 단일 모델 결과 로드

### 경로 찾기 (중요)

1. **학습 실행 후 콘솔에 출력된 경로 확인**
   - train.py 실행 시 아래처럼 출력됨:
   ```
   Results saved to a:\graduate\graduateProject\ai_service\outputs\checkpoints\20260415_231530_residualmlp\residualmlp_results.json
   ```

2. **또는 폴더에서 직접 확인**
   ```bash
   # WSL에서 확인
   ls ai_service/outputs/checkpoints/
   # 예: 20260415_231530_residualmlp  이라는 폴더가 보임
   ```

3. **폴더 이름의 구조**
   - `YYYYMMDD_HHMMSS_모델명` 형식
   - 예: `20260415_231530_residualmlp`
   - 날짜: 20260415 (2026년 4월 15일)
   - 시간: 231530 (23시 15분 30초)
   - 모델: residualmlp

### Notebook에서 경로 설정

**방법 1: 수동으로 경로 입력 (가장 간단)**

```python
# 학습 후 콘솔에서 출력된 run_name을 여기에 붙여넣기
model_run_name = "20260415_231530_residualmlp"  # 실제 폴더명으로 수정

result_path = checkpoints_root / model_run_name / "residualmlp_results.json"

with open(result_path, "r", encoding="utf-8") as f:
    result = json.load(f)

history = result["history"]
metrics = result["metrics"]
```

**방법 2: 가장 최신 결과 자동으로 찾기**

```python
from pathlib import Path

# checkpoints 폴더 내 모든 폴더를 시간 순서로 정렬
checkpoint_folders = sorted(checkpoints_root.glob("*residualmlp*"), key=lambda x: x.stat().st_mtime, reverse=True)

if checkpoint_folders:
    latest_folder = checkpoint_folders[0]
    result_path = latest_folder / "residualmlp_results.json"
    print(f"사용할 경로: {result_path}")
    
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    history = result["history"]
    metrics = result["metrics"]
else:
    print("결과 폴더를 찾을 수 없습니다.")
```

**방법 3: 모델명만 지정하고 자동 찾기**

```python
model_name = "residualmlp"  # 이것만 지정

# 해당 모델의 모든 결과 폴더를 찾음
model_folders = list(checkpoints_root.glob(f"*{model_name}"))

if model_folders:
    print("사용 가능한 실행 결과:")
    for i, folder in enumerate(sorted(model_folders, reverse=True), 1):
        print(f"{i}. {folder.name}")
    
    # 가장 최신 결과 사용
    latest_folder = sorted(model_folders, reverse=True)[0]
    result_path = latest_folder / f"{model_name}_results.json"
    
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    history = result["history"]
    metrics = result["metrics"]
else:
    print(f"{model_name} 결과를 찾을 수 없습니다.")
```

---

## 3) 학습 곡선 4종

```python
epochs = range(1, len(history["train_loss"]) + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

axes[0, 0].plot(epochs, history["train_loss"], label="train_loss")
axes[0, 0].plot(epochs, history["val_loss"], label="val_loss")
axes[0, 0].set_title("Loss")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()

axes[0, 1].plot(epochs, history["train_mae"], label="train_mae")
axes[0, 1].plot(epochs, history["val_mae"], label="val_mae")
axes[0, 1].set_title("MAE")
axes[0, 1].set_ylabel("MAE")
axes[0, 1].legend()

axes[1, 0].plot(epochs, history["train_loss"], color="tab:blue")
axes[1, 0].set_title("Train Loss")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Loss")

axes[1, 1].plot(epochs, history["val_loss"], color="tab:orange")
axes[1, 1].set_title("Validation Loss")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")

plt.tight_layout()
plt.show()
```

---

## 4) per_beta_mae 막대 그래프

```python
per_beta_mae = metrics["per_beta_mae"]
indices = np.arange(len(per_beta_mae))

plt.figure(figsize=(10, 5))
plt.bar(indices, per_beta_mae)
plt.xticks(indices, [f"beta_{i}" for i in indices], rotation=45)
plt.xlabel("Beta Index")
plt.ylabel("MAE")
plt.title("Per Beta MAE")
plt.tight_layout()
plt.show()
```

---

## 5) 예측 오차 분포 히스토그램

```python
# history/history.json에 prediction이 저장되지 않는 경우,
# test 예측값을 별도 저장한 JSON 또는 npy 파일을 사용해야 한다.
# 아래는 predictions와 targets가 이미 있다고 가정한 예시다.

# predictions = np.array(...)
# targets = np.array(...)

errors = predictions - targets
flat_errors = errors.reshape(-1)

plt.figure(figsize=(10, 5))
plt.hist(flat_errors, bins=50, alpha=0.8)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Error Distribution")
plt.tight_layout()
plt.show()
```

---

## 6) all 모드 요약 로드

```python
all_run_name = "20260415_231530_all"  # 예시
summary_path = checkpoints_root / all_run_name / "all_models_summary.json"

with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

models = summary["models"]
```

---

## 7) 모델별 지표 비교 그래프

```python
model_names = [item["model"] for item in models]
mse_values = [item["metrics"]["mse"] for item in models]
mae_values = [item["metrics"]["mae"] for item in models]
rmse_values = [item["metrics"]["rmse"] for item in models]
r2_values = [item["metrics"]["r2"] for item in models]

x = np.arange(len(model_names))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - 1.5 * width, mse_values, width, label="MSE")
plt.bar(x - 0.5 * width, mae_values, width, label="MAE")
plt.bar(x + 0.5 * width, rmse_values, width, label="RMSE")
plt.bar(x + 1.5 * width, r2_values, width, label="R2")

plt.xticks(x, model_names)
plt.xlabel("Model")
plt.ylabel("Metric Value")
plt.title("Model Comparison")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 8) 평균 per_beta_mae 비교

```python
model_names = [item["model"] for item in models]
mean_per_beta_mae = [float(np.mean(item["metrics"]["per_beta_mae"])) for item in models]

plt.figure(figsize=(10, 5))
plt.bar(model_names, mean_per_beta_mae)
plt.xlabel("Model")
plt.ylabel("Mean Per Beta MAE")
plt.title("Mean Per Beta MAE Comparison")
plt.tight_layout()
plt.show()
```

---

## 9) 결과 이미지 저장 예시

```python
save_path = visual_root / f"{model_run_name}_history.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
```

---

## 10) 사용 메모

- `train.py`는 시각화를 자동 실행하지 않는다.
- 결과 JSON 경로만 맞추면 Notebook에서 그대로 재사용 가능하다.
- 이미지 저장은 필요할 때만 수동으로 수행한다.
