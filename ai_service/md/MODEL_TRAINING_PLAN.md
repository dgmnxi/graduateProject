# 통합 학습 계획서

이 문서는 ai_service 하위의 학습 관련 계획 파일을 하나로 통합한 버전이다.
train.py는 아직 수정하지 않으며, 현재 코드 구조를 기준으로 적용 계획만 정리한다.

## 1. 현재 코드 상태 점검

- 학습 스크립트 위치: [ai_service/training/train.py](../training/train.py)
- 모델 팩토리 위치: [ai_service/training/models.py](../training/models.py)
- 현재 `--model` 인자는 `resmlp`, `gcn`만 허용됨.
- 모델 팩토리 함수는 `mlp`, `residualmlp`, `transformer`, `gcn`을 지원함.
- 즉, 현재 상태에서는 4개 모델 전체를 CLI에서 직접 학습할 수 없음.
- 추가로 `resmlp`는 팩토리의 키(`residualmlp`)와 이름이 달라 런타임 오류 가능성이 있음.
- 현재 학습 결과에는 JSON(history/metrics)만 존재하고, 시각화 자동 실행은 없음.
- 현재 저장 정책은 `./checkpoints` 기반이며, outputs 정책은 아직 미적용 상태다.

## 2. 목표

- CLI에서 아래 4개 모델을 모두 학습 가능하도록 수정:
  - `mlp`
  - `transformer`
  - `residualmlp`
  - `gcn`
- 기존 사용 습관을 고려해 `resmlp` 입력도 `residualmlp`로 자동 매핑.
- 필요 시 `--model all`로 4개 모델 연속 학습 지원.
- 학습 산출물은 `ai_service/outputs` 하위로 통일.
- 시각화는 자동 실행하지 않고, Notebook에서 복붙 가능한 코드 스니펫과 시각화 산출물을 `ai_service/outputs/visual` 하위로 저장.

## 3. 저장 경로 정책

### 3-1. 정확한 경로 기준

- 프로젝트 루트: `a:\graduate\graduateProject`
- 학습 결과 루트: `a:\graduate\graduateProject\ai_service\outputs`
- 체크포인트 루트: `a:\graduate\graduateProject\ai_service\outputs\checkpoints`
- 로그 루트: `a:\graduate\graduateProject\ai_service\outputs\logs`
- 시각화 루트: `a:\graduate\graduateProject\ai_service\outputs\visual`

즉, 저장 경로는 ai_service 하위 outputs 폴더를 기준으로 한다.

### 3-2. 실행 단위 저장 이름

- run_name 형식: `YYYYMMDD_MODEL`
  - 예: `20260415_residualmlp`

- 접미사 옵션(충돌 방지): `HHMMSS`
  - 옵션 비활성: `YYYYMMDD_MODEL`
  - 옵션 활성: `YYYYMMDD_HHMMSS_MODEL`
  - 예: `20260415_231530_residualmlp`

- 기본 권장: `HHMMSS` 자동 부여(auto)
- 필요 시 `none` 또는 `custom`을 선택하는 옵션을 계획에 포함

### 3-3. 권장 최종 구조

- `a:\graduate\graduateProject\ai_service\outputs\checkpoints\YYYYMMDD[_HHMMSS]_MODEL\`
  - `MODEL_best.pth`
  - `MODEL_results.json`

- `a:\graduate\graduateProject\ai_service\outputs\logs\`
  - `YYYYMMDD[_HHMMSS]_MODEL.log`

- all 모드 요약:
  - `a:\graduate\graduateProject\ai_service\outputs\checkpoints\YYYYMMDD[_HHMMSS]_all\all_models_summary.json`

## 4. train.py 변경 계획

### 4-1. 실행 이름(run_name) 생성

- main 진입 시 날짜 문자열 생성
- model 인자 정규화 후 run_name 생성
- all 모드일 경우 모델별 run_name을 각각 생성
- run_suffix 인자 추가를 계획에 반영
  - `auto`: HHMMSS 자동 부여
  - `none`: 날짜만 사용
  - `custom`: 사용자 지정 접미사 사용

### 4-2. 체크포인트 저장 정책 변경

현재:
- best 모델 + epoch 주기 저장(10 epoch마다)

변경:
- best 모델만 저장
- 파일 확장자는 `.pth` 사용
- OOM 발생 시에는 선택적으로 OOM 상태 JSON만 기록하고 가중치 추가 저장은 하지 않음

### 4-3. 로그 기록 체계 도입

- Python logging 모듈 사용
- 콘솔 + 파일 핸들러 동시 사용
- 로그 파일 경로:
  - `a:\graduate\graduateProject\ai_service\outputs\logs\YYYYMMDD[_HHMMSS]_MODEL.log`
- step 단위 로그는 100 step마다만 기록
  - `train/validate` 루프에서 `batch_idx % 100 == 0` 조건으로 출력
- tqdm 진행바는 유지하되 파일 로그는 100 step 간격으로 제한

### 4-4. 결과 JSON 경로 통일

- 모델별 결과 JSON:
  - `a:\graduate\graduateProject\ai_service\outputs\checkpoints\YYYYMMDD[_HHMMSS]_MODEL\MODEL_results.json`
- all 모드 요약 JSON:
  - `a:\graduate\graduateProject\ai_service\outputs\checkpoints\YYYYMMDD[_HHMMSS]_all\all_models_summary.json`

### 4-5. OOM 보호 로직

- OOM 발생 시 안전 중단 예외를 유지
- `optimizer.zero_grad(set_to_none=True)` 후 `torch.cuda.empty_cache()` 수행
- OOM 상태 checkpoint/JSON 저장 후 중단
- `main` 종료 시에도 `cleanup_cuda_memory()`가 호출되도록 유지

### 4-6. 현재 train.py 기준 예시 명령/체크리스트 유지 여부

- 모델 선택 예시 명령은 기존 구조를 유지하되, 저장 위치는 outputs 정책으로 바뀌는 것을 전제로 한다.
- 체크리스트는 삭제하지 않고, 경로 기준만 outputs 기준으로 갱신한다.
- 즉, 실행 예시와 체크리스트의 역할은 유지되고, 저장 경로만 ai_service/outputs 기준으로 명시된다.

## 5. 시각화 계획

### 5-1. 원칙

- 학습 중/학습 후 자동 시각화 실행은 하지 않음
- 시각화에 필요한 코드만 별도 파일로 저장
- 사용자가 필요할 때 Jupyter Notebook 등에서 코드를 직접 복사/실행
- 프로젝트 의존성(requirements)에는 matplotlib/seaborn 추가하지 않음

### 5-2. 구현 범위

1) 단일 모델용 코드 스니펫 저장
- history 4개 곡선: `train_loss`, `val_loss`, `train_mae`, `val_mae`
- `per_beta_mae` 막대 그래프
- 예측 오차 분포 히스토그램

2) all 모드 비교용 코드 스니펫 저장
- 모델별 `MSE/MAE/RMSE/R2` 비교
- 모델별 평균 `per_beta_mae` 비교

3) 실행 방식
- 학습 코드(train.py)와 연결하지 않음
- 시각화 코드는 독립 파일/문서로만 제공

### 5-3. 파일 구조 제안

- `ai_service/outputs/visual/VISUALIZATION_SNIPPETS.md`
  - 단일 모델 결과 JSON 로드 예시
  - all_models_summary.json 로드 예시
  - 그래프별 코드 블록(Notebook 복붙용)

- 선택: `ai_service/outputs/visual/VIS_NOTEBOOK_TEMPLATE.md`
  - Jupyter Cell 순서 템플릿
  - 1) 로드, 2) history plot, 3) metric plot, 4) 비교 plot

### 5-4. 검증 체크리스트

- [ ] train.py에 시각화 호출 코드가 없는지 확인
- [ ] requirements.txt에 matplotlib/seaborn 추가가 없는지 확인
- [ ] 스니펫 문서에 JSON 경로 변수만 바꿔 실행 가능한지 확인
- [ ] 단일 모델 결과 JSON 기준으로 history 시각화가 가능한지 확인
- [ ] all_models_summary.json 기준 비교 시각화가 가능한지 확인
- [ ] outputs 정책 경로(`ai_service/outputs/checkpoints/YYYYMMDD[_HHMMSS]_MODEL/...`) 기준으로 스니펫 경로가 동작하는지 확인
- [ ] 시각화 산출물과 스니펫이 `ai_service/outputs/visual` 하위에 저장되는지 확인

## 6. 4개 모델을 한 번에 학습할 때 예상 현상

- 장점:
  - 동일 데이터/설정으로 모델 간 성능 비교가 쉬움.
  - 실험 재현성이 좋아짐.

- 단점:
  - 총 학습 시간이 4배 수준으로 증가.
  - GPU 점유 시간이 길어져 다른 작업과 충돌 가능성 증가.
  - 체크포인트/로그 파일이 빠르게 늘어 디스크 사용량 증가.

## 7. 권장 운영 방식

- 빠른 반복/디버깅 단계: 모델을 따로 학습(단일 모델 실행).
- 최종 비교 실험 단계: `all` 모드로 일괄 학습.
- 팀 공유 보고 단계: `all_models_summary.json` 기준으로 지표 비교.
- 시각화는 Notebook에서 수동 실행.
- Notebook 복붙용 시각화 코드와 저장된 그래프는 `ai_service/outputs/visual` 하위에 관리.

## 8. 실행 후 확인 체크리스트

### A. 실행 전
- [ ] `ai_service/outputs/checkpoints` 경로가 기준인지 확인
- [ ] `ai_service/outputs/logs` 경로가 기준인지 확인
- [ ] `data/processed/*.npz` 3종 파일 존재 확인

### B. 실행 중
- [ ] 콘솔에 선택한 모델명(또는 all 모드의 각 모델명) 출력 확인
- [ ] Epoch 진행바(Training/Validation) 정상 갱신 확인
- [ ] Loss/MAE 값이 NaN 없이 출력되는지 확인
- [ ] 로그가 100 step 간격으로만 기록되는지 확인

### C. 실행 후 파일
- [ ] 체크포인트 파일 생성 확인(`MODEL_best.pth`)
- [ ] 결과 JSON 생성 확인(`MODEL_results.json`)
- [ ] all 모드 사용 시 `all_models_summary.json` 생성 확인
- [ ] 로그 파일 생성 확인(`YYYYMMDD[_HHMMSS]_MODEL.log`)

### D. 실행 후 지표
- [ ] MSE/MAE/RMSE/R2가 출력되는지 확인
- [ ] `per_beta_mae` 길이가 10인지 확인
- [ ] 모델 간 지표 비교 시 동일 데이터셋 기준인지 확인

## 9. 예시 명령

아래 값은 현재 코드 구조 기준의 시작점(추천값)이다. 데이터 크기/GPU 메모리에 맞춰 조정한다.

1. MLP (빠른 베이스라인)

```bash
python ai_service/training/train.py --model mlp --data_dir ./data/processed --epochs 80 --batch_size 64 --lr 0.001 --weight_decay 0.0001 --save_dir ./checkpoints
```

2. ResidualMLP (권장 기본 모델)

```bash
python ai_service/training/train.py --model residualmlp --data_dir ./data/processed --epochs 100 --batch_size 32 --lr 0.001 --weight_decay 0.0001 --save_dir ./checkpoints
```

3. Transformer (메모리/시간 부담 큼)

```bash
python ai_service/training/train.py --model transformer --data_dir ./data/processed --epochs 120 --batch_size 16 --lr 0.0005 --weight_decay 0.0001 --save_dir ./checkpoints
```

4. GCN (관절 그래프 구조 반영)

```bash
python ai_service/training/train.py --model gcn --data_dir ./data/processed --epochs 120 --batch_size 32 --lr 0.0008 --weight_decay 0.0001 --save_dir ./checkpoints
```

5. 전체 모델 일괄 비교 실행

```bash
python ai_service/training/train.py --model all --data_dir ./data/processed --epochs 100 --batch_size 32 --lr 0.001 --weight_decay 0.0001 --save_dir ./checkpoints
```

6. `resmlp` 별칭 사용 (동일 동작)

```bash
python ai_service/training/train.py --model resmlp --data_dir ./data/processed --epochs 100 --batch_size 32
```

## 10. 마이그레이션/호환

- 기존 `--save_dir` 인자는 하위 호환을 위해 유지하되, 기본 동작은 outputs 정책을 우선 적용한다.
- 과거 `checkpoints` 경로를 읽는 스크립트가 있다면 새 경로로 업데이트 필요하다.
- 현재 train.py의 기본 `save_dir('./checkpoints')`는 정책 적용 시 `ai_service/outputs/checkpoints`로 매핑하는 방향을 계획한다.

## 11. 리스크 및 대응

- 완료
  - 리스크: all 모드에서 로그 파일이 섞일 수 있음
  - 대응(적용 필수): 모델별 logger 이름 및 파일 핸들러를 분리하고, 모델 종료 시 핸들러를 반드시 정리
  - train.py 반영 근거:
    - `setup_logger(run_name, ...)`로 모델 실행 단위 logger를 분리.
    - `run_name`에 모델명이 포함되어 로그 파일이 모델별로 분리됨.
    - `train_and_evaluate_single_model`의 `finally`에서 핸들러 flush/close/remove 수행.

- 완료
  - 리스크: 날짜만 사용 시 같은 날 재실행 충돌 가능
  - 대응(적용 필수): HHMMSS 접미사 옵션을 기본 활성화(auto)로 제공하고, 필요 시 none/custom 선택 허용
  - train.py 반영 근거:
    - `--run_suffix` 기본값이 `auto`이며 선택지가 `auto|none|custom`으로 제공됨.
    - `build_run_name`에서 기본적으로 시각(YYYYMMDD_HHMMSS)을 포함한 run_name 생성.

- 완료
  - 리스크: OOM 발생 시 중간 상태가 소실될 수 있음
  - 대응(적용 필수): OOM 체크포인트/JSON 저장 후 안전 종료, cleanup_cuda_memory() 유지
  - train.py 반영 근거:
    - `Trainer.train`에서 OOM 시 `*_oom_epoch{epoch}.pth` 저장 후 예외 재전파.
    - `train_and_evaluate_single_model`에서 OOM 상태(`failed_oom`)와 에러 메시지를 결과 JSON에 기록.
    - 모델 학습 단위 `finally`와 `main`의 `finally`에서 `cleanup_cuda_memory()` 호출 유지.

- 완료
  - 리스크: 시각화 의존성을 학습 환경에 넣으면 재현성이 흔들릴 수 있음
  - 대응(적용 필수): 시각화는 Notebook용 문서만 제공하고 requirements에는 추가하지 않음
  - train.py 반영 근거:
    - train.py 내부에 시각화 라이브러리 import 및 시각화 실행 로직이 없음.
    - 학습 코드가 시각화와 분리되어 동작함.

- 미완료
  - 리스크: 기존 결과 탐색 스크립트 경로 불일치
  - 대응(적용 필수): 경로 변경 공지 + 기존 경로를 새 경로로 바꾸는 변환 스크립트 제공
  - train.py 기준 판단 근거:
    - 학습 출력 경로 정책은 반영되어 있으나, 경로 변경 공지 로직은 코드상 존재하지 않음.
    - 기존 경로를 새 경로로 변환하는 별도 스크립트 제공은 train.py 내에서 확인되지 않음.

## 12. 적용 항목 문서화

- 코드 적용 전/후 확인 항목은 본 문서의 8절(실행 후 확인 체크리스트)과 11절(리스크 및 대응)에 통합해 관리한다.
- 별도 적용 항목 문서는 만들지 않는다.

