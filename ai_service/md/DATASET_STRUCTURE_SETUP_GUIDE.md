# 데이터셋 구조 설정 가이드 (AGORA + 3DPW)

이 문서는 프로젝트에서 AGORA, 3DPW 데이터셋을 일관되게 보관하고 사용하는 방법을 설명합니다.

## 1. 왜 폴더를 나누나?

AGORA와 3DPW를 분리하면 다음이 쉬워집니다.

- 데이터 출처별 관리
- pretrain(AGORA) / finetune(3DPW) 실험 분리
- 경로 실수 감소
- 팀 작업 시 충돌 감소

---

## 2. 권장 폴더 구조

프로젝트 루트 기준으로 아래 구조를 사용합니다.

```text
data/
  raw/
    agora/
      images/
      annotations/
      splits/
      README.md
    3dpw/
      images/
      annotations/
      splits/
      README.md
  processed/
```

### 각 폴더 의미

- `images/`: 원본 이미지
- `annotations/`: 라벨 파일(pkl/json/csv 등)
- `splits/`: train/val/test 분할 정보
- `processed/`: 전처리 완료 결과(npz 등)

중요: `raw`는 원본 보관용입니다. 원본 파일을 직접 수정하지 마세요.

---

## 3. 지금 이미 생성된 폴더

아래 폴더는 이미 생성되어 있습니다.

- `data/raw/agora/images`
- `data/raw/agora/annotations`
- `data/raw/agora/splits`
- `data/raw/3dpw/images`
- `data/raw/3dpw/annotations`
- `data/raw/3dpw/splits`

---

## 4. 실제 데이터 옮기는 순서

1. AGORA 이미지를 `data/raw/agora/images`로 이동
2. AGORA 라벨 파일을 `data/raw/agora/annotations`로 이동
3. 3DPW 이미지를 `data/raw/3dpw/images`로 이동
4. 3DPW 라벨 파일을 `data/raw/3dpw/annotations`로 이동
5. split 파일(train/val/test)을 각 `splits` 폴더에 저장

---

## 5. 코드에서 경로 쓰는 방법

`ai_service/config.py`에 데이터셋 경로 템플릿이 추가되어 있습니다.

### 추가된 주요 경로 상수

- `DATA_ROOT`
- `RAW_DATA_ROOT`
- `PROCESSED_DATA_ROOT`
- `AGORA_ROOT`, `AGORA_IMAGES_DIR`, `AGORA_ANNOTATIONS_DIR`, `AGORA_SPLITS_DIR`
- `D3PW_ROOT`, `D3PW_IMAGES_DIR`, `D3PW_ANNOTATIONS_DIR`, `D3PW_SPLITS_DIR`

### 예시

```python
from ai_service.config import get_config

cfg = get_config()

print(cfg.AGORA_IMAGES_DIR)
print(cfg.D3PW_ANNOTATIONS_DIR)
```

---

## 6. 환경변수로 루트 경로 바꾸기 (선택)

기본값은 프로젝트 내부 `data` 폴더입니다.
다른 디스크를 쓰고 싶으면 `DATA_ROOT`를 설정하세요.

### PowerShell 예시

```powershell
$env:DATA_ROOT = "D:/dataset_storage/graduate_data"
```

이후 코드에서 `Config.DATA_ROOT`는 위 경로를 사용합니다.

---

## 7. 초보자 체크리스트

- [ ] AGORA/3DPW를 `raw` 아래 분리 저장했는가?
- [ ] 이미지와 라벨을 `images`, `annotations`로 구분했는가?
- [ ] split 파일을 `splits`에 저장했는가?
- [ ] 전처리 결과를 `processed`에 저장하도록 했는가?
- [ ] 코드에서 경로를 하드코딩하지 않고 `config.py`를 사용했는가?

---

## 8. 자주 하는 실수

1. `raw` 데이터를 전처리로 덮어쓰기
2. AGORA와 3DPW 라벨 파일을 같은 폴더에 섞어두기
3. 코드에 절대경로를 하드코딩하기
4. split 기준(train/val/test)을 문서화하지 않기

---

## 9. 다음 권장 작업

- `prepare_data.py`를 AGORA/3DPW 분리 경로 기반으로 리팩터링
- AGORA pretrain, 3DPW finetune 실행 스크립트 분리
- 실험 로그에서 pretrain 유무를 명확히 구분 저장
