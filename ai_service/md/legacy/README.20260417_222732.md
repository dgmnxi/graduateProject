# 의류 추천 AI 서비스

체형에 따른 의류 추천을 위한 AI 파트 구현

## 현재 상태 vs 목표 상태

- 현재 구현 상태:
  - API 입력이 `joints` 중심이며, Stage 1은 `services/pose_to_beta.py` 인터페이스만 존재
  - 체형 추정 모델은 아직 실제 추론 모델로 연결되지 않음
- 목표 상태 (서비스 요구사항):
  - 사용자가 전신 사진을 업로드
  - 서버가 필요한 영역만 마스킹/전처리
  - AI가 체형 정보를 추출해 beta 또는 체형 토큰 생성
  - Stage 2-3에서 프롬프트/토큰화 후 Marqo 검색으로 추천 반환

즉, 현재 구조는 "완전히 반영" 상태가 아니라 "초기 골격" 상태입니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│ 백엔드 서버 (FastAPI)                                        │
│                                                              │
│ /recommend POST 요청                                        │
│   ├─ user_id: 사용자 ID                                     │
│   ├─ masked_fullbody_image: 마스킹된 전신 이미지             │
│   ├─ tags: 의류 추천 태그                                   │
│   └─ metadata: 추가 정보                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │  Stage 1: Image→Beta │
         │                      │
         │ 마스킹 전신 이미지    │
         │        ↓             │
         │  사전학습 인체 모델  │ (HMR2/PARE/CLIFF)
         │        ↓             │
         │   beta값 (10개)      │
         └────────┬─────────────┘
                  │
         ┌────────▼──────────┐
         │ Stage 2-3: 프롬프트 생성 │
         │                   │
         │ beta값 + 태그     │
         │        ↓          │
         │ 프롬프트 생성     │
         │ (체형+선호도)     │
         └────────┬──────────┘
                  │
         ┌────────▼─────────────────┐
         │ Stage 4: Marqo 검색      │
         │                          │
         │ SigLiP 임베딩 프롬프트   │
         │        ↓                 │
         │  Marqo 벡터 검색         │ /// 설치 필요
         │        ↓                 │
         │  추천 의류 반환           │
         └────────┬──────────────────┘
                  │
         ┌────────▼─────────────┐
         │ 최종 결과 반환        │
         │                      │
         │ - 추천 의류           │
         │ - 체형 정보           │
         │ - 생성 프롬프트       │
         └──────────────────────┘
```

## 디렉토리 구조

```
ai_service/
├── main.py                 # FastAPI 메인 애플리케이션
├── config.py               # 설정 파일
├── requirements.txt        # Python 의존성
├── .env.example            # 환경변수 예제
├── models/
│   └── schemas.py          # Pydantic 데이터 모델
└── services/
    ├── pose_to_beta.py     # Stage 1: 관절값 → beta
    ├── profile_generator.py # Stage 2-3: 프롬프트 생성
    └── marqo_service.py    # Stage 4: Marqo 검색
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 현재 디렉토리: c:\Users\Min\Desktop\graduateProject\ai_service

# 환경변수 파일 생성
cp .env.example .env

# 필요시 .env 파일 수정
```

### 2. Python 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3. 서버 실행

```bash
# 개발 모드 (자동 재로드)
python main.py

# 또는
uvicorn main:app --reload
```

서버는 `http://localhost:8001`에서 시작됩니다.

## API 사용법

> 참고: 아래 `joints` 기반 요청은 현재 구현 호환을 위한 레거시 형태입니다. 목표 서비스는 `masked_fullbody_image` 입력 기반으로 전환합니다.

### 기본 추천 요청

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "joints": [0.1, 0.2, 0.3, ...],  # 33개 또는 필요한 관절값
    "tags": {
      "season": "봄",
      "style": "캐주얼",
      "color": "파란색",
      "category": "상의",
      "additional_tags": ["활동적인"]
    }
  }'
```

### 목표 API 요청 (마스킹 이미지 기반)

```bash
curl -X POST "http://localhost:8001/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "masked_fullbody_image": "<base64-encoded-image>",
    "tags": {
      "season": "봄",
      "style": "캐주얼",
      "color": "파란색",
      "category": "상의",
      "additional_tags": ["활동적인"]
    },
    "metadata": {
      "capture_distance_m": 2.5,
      "camera_height_m": 1.2,
      "pose": "front_standing"
    }
  }'
```

### 응답 예제

```json
{
  "user_id": "user123",
  "recommendations": [
    {
      "id": "item_1",
      "title": "추천 의류 1",
      "description": "...",
      "image_url": "...",
      "tags": ["..."],
      "score": 0.95
    }
  ],
  "profile": {
    "user_id": "user123",
    "beta_values": [0.1, -0.2, ...],
    "tags": {...},
    "generated_prompt": "다음 조건에 맞는 의류를 찾아주세요: ...",
    "prompt_components": {...}
  }
}
```

## 단계별 상세 설명

### Stage 1: 마스킹 이미지 → Beta 값

**파일**: `services/pose_to_beta.py`

- 입력: 마스킹된 전신 이미지
- 전처리: 인체 영역 정렬, 리사이즈, 정규화
- 추론: 사전학습 인체 모델(HMR2/PARE/CLIFF 계열)로 beta 추정
- 출력: beta(10), confidence, 품질 메타데이터
- **현재 상태**: 관절 기반 인터페이스만 존재하며, 이미지 기반 추론 경로는 구현 예정

**확장 포인트**:
```python
# models/your_model.py에서 모델 준비 후
model = torch.load('path/to/model.pth')
converter.set_model(model)
```

### Stage 2-3: Beta값 + 태그 → 프롬프트

**파일**: `services/profile_generator.py`

1. Beta 값 해석:
   - beta[0]: 전체 크기
   - beta[1]: 길쭉함/뚱뚱함
   - beta[2-9]: 세부 체형 특징

2. 태그 키워드 추출:
   - 계절별 키워드 매핑
   - 스타일별 키워드 매핑
   - 색상, 부위 태그 추가

3. 프롬프트 생성:
   - 템플릿 기반 프롬프트 구성
   - 체형 + 스타일 정보 통합

이 단계는 목표 요구사항과 정합성이 높으며, Stage 1 입력 전환 후 그대로 활용 가능합니다.

### Stage 4: Marqo 검색

**파일**: `services/marqo_service.py`

- SigLiP 임베딩 기반 벡터 검색
- 프롬프트와 유사한 의류 검색
- 필터링 옵션 지원 (색상, 가격 등)

**설정 필요**:
```bash
# 먼저 Marqo 설치 및 실행
pip install marqo

# Marqo 서버 시작
marqo run --help
```

## 디버그 API

각 단계별 결과를 확인할 수 있는 디버그 API:

```bash
# Stage 1 테스트
curl -X POST "http://localhost:8001/debug/pose-to-beta" ...

# Stage 2-3 테스트
curl -X POST "http://localhost:8001/debug/prompt-generation" ...

# Stage 4 테스트
curl -X POST "http://localhost:8001/debug/marqo-search?prompt=..." ...
```

향후 추가 권장:

```bash
# 이미지 기반 Stage 1 테스트 (목표)
curl -X POST "http://localhost:8001/debug/image-to-beta" ...
```

## 촬영/입력 가이드 (서비스 요구사항)

체형 추론 안정성을 위해 사용자 촬영 조건을 API 요구사항으로 둘 수 있습니다.

필수 권장 조건:
- 전신이 프레임 안에 모두 포함될 것 (머리~발끝)
- 카메라-사용자 거리: 2.0m ~ 3.5m
- 카메라 높이: 골반~가슴 높이 권장 (과도한 하이/로우 앵글 금지)
- 정면 자세, 팔은 몸에서 약간 떨어뜨린 자연 기립
- 헐렁한 외투/가방/큰 가림 요소 최소화
- 밝은 배경 대비, 인물 윤곽이 분명한 조명

품질 필터(서버 측 reject 예시):
- 인체 bbox가 프레임의 40% 미만 또는 95% 초과
- 발/머리 landmark 누락
- 심한 모션블러 또는 저해상도

## 환경변수

`.env` 파일에서 설정:

| 변수 | 기본값 | 설명 |
|------|--------|------|
| HOST | 0.0.0.0 | 서버 호스트 |
| PORT | 8001 | 서버 포트 |
| DEBUG | False | 디버그 모드 |
| POSE_MODEL_PATH | None | AI 모델 경로 |
| MARQO_URL | http://localhost:8000 | Marqo 서버 URL |
| CORS_ORIGINS | http://localhost:3000 | CORS 허용 출처 |

## 다음 단계

### 1. AI 모델 구축 (Stage 1)
- [ ] 이미지 기반 입력 스키마 추가 (`masked_fullbody_image`)
- [ ] 사전학습 인체 모델(HMR2/PARE/CLIFF) 1종 연결
- [ ] 모델 경로를 `config.POSE_MODEL_PATH`에 설정
- [ ] Stage 1을 이미지 기반 추론으로 전환
- [ ] 촬영 품질 체크(거리/구도/가림) 메타데이터 검증 추가

### 2. Marqo 통합 (Stage 4)
- [ ] Marqo 설치 및 실행
- [ ] 의류 데이터베이스 인덱싱
- [ ] `services/marqo_service.py` 실제 구현

### 3. 프롬프트 최적화 (Stage 2-3)
- [ ] 템플릿 개선
- [ ] 태그 매핑 확장
- [ ] 피드백 기반 튜닝

### 4. 테스트 및 배포
- [ ] 단위 테스트 작성
- [ ] E2E 테스트
- [ ] Docker 컨테이너화
- [ ] 성능 최적화

## 주요 기술 스택

- **프레임워크**: FastAPI
- **데이터 검증**: Pydantic
- **벡터 검색**: Marqo (SigLiP)
- **비전 모델**: 사전학습 인체 모델(HMR2/PARE/CLIFF), MediaPipe는 보조(검출/품질체크)
- **체형 모델**: SMPL
- **배포**: Uvicorn

## 참고사항

1. **Stage 1 전환 필요**: 현재 `joints` 중심 인터페이스를 `masked_fullbody_image` 중심으로 전환해야 목표 요구사항과 일치합니다.
2. **Marqo 설치 필요**: Stage 4를 실제로 사용하려면 Marqo를 별도로 설치해야 합니다.
3. **태그 형식 확정 예정**: 현재 예상 태그로만 구현했습니다.
4. **모의 데이터**: 현재 각 단계에 모의 데이터가 포함되어 있어 즉시 테스트 가능합니다.

## 문제 해결

### Marqo 연결 실패
```
⚠ Marqo check failed - will use mock results
```
→ Marqo 서버가 실행 중인지 확인하세요.

### 모델 로드 실패
```
✗ Pose to Beta converter initialization failed
```
→ `POSE_MODEL_PATH`가 올바른지 확인하세요.

## 라이선스

[프로젝트 라이선스]

## 기여

[기여 가이드]
