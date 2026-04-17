# 3DPW 데이터셋 상세 분석

## 1) 한눈에 요약

- 정식 명칭: 3D Poses in the Wild (3DPW)
- 대표 논문: Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera (ECCV 2018)
- 핵심 특징:
  - 야외/실생활 장면에서 획득된 3D 사람 자세 데이터
  - 고정 스튜디오가 아닌 이동식 핸드헬드 카메라 기반
  - IMU와 영상 정보를 결합해 비교적 정확한 3D 참조 자세 제공
- 공식 소개 기준 주요 규모:
  - 60개 비디오 시퀀스
  - 프레임별 카메라 포즈
  - 2D 포즈 주석
  - 3D 포즈/신체 모델(SMPL 기반) 관련 정보

이 데이터셋은 "실험실 중심" 데이터(Human3.6M 등) 대비 실제 환경 일반화 성능을 확인하기에 매우 유용합니다.

---

## 2) 왜 3DPW가 중요한가

3D 인체 자세 추정 연구에서 가장 어려운 부분은 다음 두 가지입니다.

- 실환경 복잡도: 가림(occlusion), 조명 변화, 다양한 배경/카메라 움직임
- 정답(ground truth) 확보: 야외에서 정확한 3D 라벨을 만들기 어려움

3DPW는 IMU + 영상 기반 정합으로 이 문제를 완화하여, "실환경에서의 정량 평가"를 가능하게 한 대표 벤치마크입니다.

---

## 3) 데이터 수집 및 라벨 생성 방식

## 3.1 촬영 방식

- 핸드헬드 카메라(이동 카메라)로 장면 촬영
- 피험자는 IMU를 착용하고 일상 동작 수행
- 결과적으로 카메라와 인물이 동시에 움직이는 어려운 조건 포함

## 3.2 라벨 생성 개념

- IMU 신호 + 영상 정보를 결합해 3D 자세를 추정
- 프레임 단위로 사람의 2D/3D 정보와 카메라 상태를 연결
- SMPL 기반 파라미터(자세/형상)를 통해 일관된 인체 표현 제공

참고: 3DPW에서 "ground truth"라는 표현은 센서 융합 기반의 참조 정답(reference) 의미이며, 절대 오차 0을 의미하는 계측실 완전 정답과는 다릅니다.

---

## 4) 포함 주석(Annotation)과 연구 활용도

공식 설명 기준으로 다음 정보가 핵심입니다.

- 2D 포즈
- 3D 포즈
- 프레임별 카메라 포즈
- 3D 신체 모델 관련 정보(재포즈/리셰이프 가능한 사람 모델)

연구 활용 관점:

- 단안 3D 자세 추정
- 다중 인물 장면에서의 포즈 추정/추적
- 카메라 움직임이 있는 환경에서의 강건성 평가
- 포즈뿐 아니라 shape(형상) 예측 품질 평가

---

## 5) 3DPW 평가 프로토콜(공식 Evaluation 페이지 요약)

3DPW는 sequenceFiles.zip 내 시퀀스를 train/validation/test로 분할해 제공하며, 비교 가능성을 위해 프로토콜 명시를 권장합니다.

## 5.1 Protocol

- All-Test-mode:
  - train/validation/test 전부를 테스트 용도로 사용
- Train-Test-mode:
  - train으로 학습, validation으로 검증, test로 최종 리포트
- Validation-mode:
  - validation은 검증 전용(학습 금지)
  - train/test를 테스트 용도로 활용 가능
- All-Train-mode:
  - 3DPW를 학습 전용으로 사용하고, 평가는 다른 데이터에서 수행

논문/보고서에서는 반드시 어떤 프로토콜인지 명시하는 것이 재현성과 공정 비교에 중요합니다.

## 5.2 Metric

공식 페이지에서 권장하는 핵심 지표:

- Joint error metric:
  - 예측 관절과 SMPL 관절 간 평균 유클리드 거리
- Mesh error metric:
  - 예측 메시와 GT 메시 간 평균 거리
- Mesh error metric (unposed):
  - 포즈를 0으로 정규화한 공간에서 메시 오차(형상 품질 분리 평가)
- Orientation error metric:
  - 부위 회전의 지오데식 거리

또한 Procrustes 정렬 적용/미적용 결과를 모두 보고하는 것을 권장합니다.

---

## 6) 3DPW의 강점과 한계

## 6.1 강점

- 실제 환경 다양성: 실내 고정 환경 편향을 줄임
- 카메라 이동 포함: 실사용 시나리오와 유사
- 포즈 + 형상 + 카메라 정보: 다목적 평가 가능
- 커뮤니티 표준 벤치마크로 활용 이력 풍부

## 6.2 한계

- 실환경 특성상 가림/블러/원거리 인물에서 라벨 불확실성 증가 가능
- 데이터 편중(장면, 인물 구성, 동작 빈도) 가능성
- 일부 연구는 서로 다른 전처리/평가 설정을 사용하여 직접 비교가 어려울 수 있음
- 라이선스 제약으로 재배포/2차 배포에 제한이 있을 수 있음

---

## 7) 현재 프로젝트 데이터와의 연결 분석

현재 프로젝트의 처리 결과([data/data.md](data/data.md))를 기준으로 보면:

- 입력: MediaPipe 33개 관절 x/y/z = 99차원
- 타깃: SMPL betas 10차원
- 분할: train/validation/test NPZ 파일로 구성

즉, 원본 3DPW의 풍부한 정보 중에서 본 프로젝트는

- 관절 기반 입력 표현(99D)
- 형상 파라미터(beta) 회귀

라는 명확한 supervised regression 문제로 축소해 사용하고 있습니다.

이 설계의 장점:

- 학습 목표가 단순하고 빠르게 실험 가능
- 모델 구조 비교(MLP/Transformer/GCN 등)에 집중하기 좋음

주의점:

- 2D->3D 추정 오차(MediaPipe 오검출/가림)가 betas 회귀 성능에 직접 전파
- 포즈 변화가 큰 샘플에서 betas 식별성이 약해질 수 있음
- test split의 betas 분포가 train/val과 다르면 일반화 성능 왜곡 가능

---

## 8) 실무용 점검 체크리스트

아래 순서대로 점검하면 초보자도 재현성 있는 실험을 구성할 수 있습니다.

1. 라이선스 확인
   - 3DPW 라이선스에서 허용하는 사용 범위(연구/비상업/재배포 제한) 확인
2. 프로토콜 명시
   - Train-Test-mode 등 사용 모드를 실험 로그와 리포트에 고정
3. 입력 정규화 검증
   - 관절 좌표 정규화 기준(카메라 좌표계/루트 정렬 여부) 일관성 확인
4. 분할 누수 방지
   - 같은 시퀀스/유사 프레임이 train-test에 동시에 들어가지 않도록 점검
5. 지표 다각화
   - beta MSE 외에, 가능하면 메시/관절 오차로 다운스트림 품질도 확인
6. 분포 점검
   - split별 betas 평균/분산/극단값을 비교해 분포 불일치 조기 탐지
7. 실패 케이스 분석
   - 가림, 원거리, 측면 자세 등 난이도별 성능 분해 리포트 작성

---

## 9) 본 프로젝트에 권장되는 추가 분석 항목

다음 분석을 붙이면 3DPW 기반 실험의 설득력이 크게 올라갑니다.

- split별 beta 차원(10개) 히스토그램/박스플롯
- 동작 유형별(걷기/달리기/상호작용) 오차 분해
- 관절 신뢰도(visibility 유사 지표) 기준 성능 곡선
- 모델별(Mlp/ResidualMlp/Transformer/Gcn) 오차 상관 분석
- out-of-distribution 샘플 탐지(마할라노비스 거리 등)

---

## 10) 인용 및 공식 링크

- 3DPW 메인 페이지: https://virtualhumans.mpi-inf.mpg.de/3DPW/
- 3DPW Evaluation: https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html
- 3DPW Challenge: http://virtualhumans.mpi-inf.mpg.de/3DPW_Challenge/
- ECCV 2018 논문(공식 인용 대상):
  - von Marcard et al., Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera

---

## 부록) 현재 프로젝트 데이터 요약 참조

- 현재 NPZ 기반 통계/구조는 [data/data.md](data/data.md) 참고
- 이 문서는 "원본 3DPW 자체"의 특성과 평가 관점을 정리한 문서
