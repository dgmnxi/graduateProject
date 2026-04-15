# 모델 학습 확장 계획서

## 1) 현재 코드 상태 점검

- 학습 스크립트 인자 `--model`은 현재 `resmlp`, `gcn`만 허용됨.
- 모델 팩토리 함수는 `mlp`, `residualmlp`, `transformer`, `gcn`을 지원함.
- 즉, 현재 상태에서는 4개 모델 전체를 CLI에서 직접 학습할 수 없음.
- 추가로 `resmlp`는 팩토리의 키(`residualmlp`)와 이름이 달라 런타임 오류 가능성이 있음.

## 2) 목표

- CLI에서 아래 4개 모델을 모두 학습 가능하도록 수정:
  - `mlp`
  - `transformer`
  - `residualmlp`
  - `gcn`
- 기존 사용 습관을 고려해 `resmlp` 입력도 `residualmlp`로 자동 매핑.
- 필요 시 `--model all`로 4개 모델 연속 학습 지원.

## 3) 구현 계획

1. `train.py`의 인자 파서를 확장한다.
- `--model` choices에 `mlp`, `transformer`, `residualmlp`, `gcn`, `resmlp`, `all` 추가.

2. 모델 이름 정규화 함수를 추가한다.
- `resmlp` -> `residualmlp` 자동 변환.

3. 단일 모델 학습 함수를 분리한다.
- 모델 생성, 학습, 평가, JSON 저장 흐름을 함수로 캡슐화.

4. `all` 모드를 추가한다.
- 4개 모델을 순차 학습하고 모델별 폴더에 체크포인트/결과 저장.
- 전체 요약 파일(`all_models_summary.json`) 저장.
- GPU 사용 시 모델 사이 캐시 정리 수행.

## 4) 4개 모델을 한 번에 학습할 때 예상 현상

- 장점:
  - 동일 데이터/설정으로 모델 간 성능 비교가 쉬움.
  - 실험 재현성이 좋아짐.

- 단점:
  - 총 학습 시간이 4배 수준으로 증가.
  - GPU 점유 시간이 길어져 다른 작업과 충돌 가능성 증가.
  - 체크포인트/로그 파일이 빠르게 늘어 디스크 사용량 증가.

## 5) 권장 운영 방식

- 빠른 반복/디버깅 단계: 모델을 따로 학습(단일 모델 실행).
- 최종 비교 실험 단계: `all` 모드로 일괄 학습.
- 팀 공유 보고 단계: `all_models_summary.json` 기준으로 지표 비교.

## 6) 실행 후 확인 체크리스트


### A. 실행 전
- [ ] `data/processed/*.npz` 3종 파일 존재 확인

### A. 실행 중
- [ ] 콘솔에 선택한 모델명(또는 all 모드의 각 모델명) 출력 확인
- [ ] Epoch 진행바(Training/Validation) 정상 갱신 확인
- [ ] Loss/MAE 값이 NaN 없이 출력되는지 확인

### B. 실행 후 파일
- [ ] 체크포인트 파일 생성 확인 (`*_best.pt`, `*_epoch10.pt` ...)
- [ ] 결과 JSON 생성 확인 (`*_results.json`)
- [ ] all 모드 사용 시 `all_models_summary.json` 생성 확인

### C. 실행 후 지표
- [ ] MSE/MAE/RMSE/R2가 출력되는지 확인
- [ ] `per_beta_mae` 길이가 10인지 확인
- [ ] 모델 간 지표 비교 시 동일 데이터셋 기준인지 확인

## 7) 예시 명령

아래 값은 현재 코드 구조 기준의 시작점(추천값)입니다. 데이터 크기/GPU 메모리에 맞춰 조정합니다.

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
