# aims_classifier
## 프로젝트 개요

### 문제 정의

#### 문제점
- 기존의 문서 분류기는 OCR을 통해 키워드를 확인하는 방식이나, 그 키워드가 매우 단순하여 실제 해당 문서가 아니여도 그런 것처럼 분류될 수 있었다.
- 또한, OCR 상태가 좋지 않으면 필요한 키워드가 누락될 수도 있었다.
- 현재 시스템은 전혀 다른 문서라도 키워드만 포함한다면 기준을 충족한다고 잘못 여길 수 있는 논리적 구멍이 있다.

#### 제약 사항
- 데이터셋이 매우 적다.(6개 클래스, 클래스당 7~9개로 총 47장)
- OCR 데이터 활용 시 별도의 전처리가 필요하다.
- 웹에서 실시간 추론이 가능해야 하므로 매우 단순한 방식으로 해결해야 한다.
- 개발 시간이 부족하다(최장 4일).

### 해결 방향

#### 핵심 목표
- 가벼운 모델 개발하기 위해 지식 증류를 적용한다.
- 분류 성능은 90% 이상을 목표로 한다.

#### 데이터셋 문제
- 서류가 여러 장이더라도 서류 판별에는 맨 앞 장이면 충분하다.
- 프로덕트에 적용할 때는 pdf 입력이 들어오므로  추론 시에는 pdf -> img 전처리가 필요하다.


## 모델 개발

### 데이터셋
#### 학습 데이터
- 웹에 공개되어 있는 이미지를 수집하였다.
- 레이아웃만 학습하면 되기 때문에 개인 정보를 가리기 위해 박스 처리한 것을 그대로 사용하였다.
- jpg 혹은 png 형식의 이미지 49장
    - no01 : 검정고시합격증명서(9장)
    - no02 : 국민체력100(8장)
    - no03 : 기초생활수급자증명서(7장)
    - no04 : 주민등록초본(9장)
    - no05 : 체력평가(8장)
    - no06 : 생활기록부대체양식(8장)
#### 테스트 데이터
- 각 클래스 당 2~3장, 총 11장

### 모델 선정
- 데이터셋이 매우 적기 때문에 적은 데이터셋으로 높은 성능을 내기 위해 사전 학습 모델을 사용한다.

#### Teacher : EfficientNet-B0
- EfficientNet은 전이 학습 효과가 뛰어나 적은 데이터로 높은 성능을 낼 수 있다.
    - 문서 분류는 일반적으로 민감 정보를 포함하고 있고 양식이 너무 다양하여 범용적인 데이터셋 구축이 어려워 데이터셋이 적은 경우가 많다.
    - 이미 ImageNet과 같은 대량 데이터를 학습하였기 때문에 이를 활용하여 적은 데이터만으로 빠르게 학습할 수 있다.
- Compound Scaling 기법을 사용하여 너비, 깊이, 해상도를 동시에 조정하여 문서 내 요소 간 관계를 잘 파악한다.
- 문서 레이아웃 학습이 필요한데, CNN은 공간적 구조를 잘 학습한다.
- CNN은 구조가 규칙적이고 연산이 단순하기 때문에 웹에서의 실시간 추론을 위한 ONNX 변환 시 네이티브 연산으로 변환이 쉽다.(ONNX가 기본적으로 Conv 연산을 지원)
- 모바일 환경 등에 최적화되어 있고 하드웨어 가속 지원이 풍부하여 변환이 쉽고 변환 후 성능 저하가 적다.

#### Student : MobileNetV3-Small
- 가장 가벼운 모델이다.

### 모델 학습

#### 평가 지표
- 정확도

#### 프로젝트 구조
```
    aims_image_classifier/               # 프로젝트 루트 디렉터리
    │── data/                             # 데이터 관련 디렉터리
    │   ├── datasets/                     # 데이터셋 관련 코드
    │   │   ├── dataloader.py             # DataLoader 정의
    │   │   ├── dataset.py                # 커스텀 데이터셋 클래스 정의
    │   │   ├── preprocess.py             # 데이터 전처리 함수
    │   ├── fonts/                        # 폰트 파일 저장 폴더 (OCR 등에 사용될 수 있음)
    │── model_pth/                        # 학습된 모델 가중치 저장 디렉터리
    │── models/                           # 모델 아키텍처 정의
    │   ├── student_models.py             # Student 모델 정의 (MobileNet 등)
    │   ├── teacher_eff.py                # Teacher 모델 정의 (EfficientNet 등)
    │── utils/                            # 유틸리티 함수 및 도구 모음
    │   ├── eda/                          # 데이터 탐색 (EDA) 관련 코드
    │   ├── config.py                     # 설정값 및 하이퍼파라미터 관리
    │   ├── convert_onnx.py               # 모델을 ONNX 형식으로 변환하는 코드
    │   ├── early_stopping.py             # Early Stopping 구현
    │   ├── gpu_utils.py                  # GPU 관련 유틸리티 함수
    │   ├── kfold_merger.py               # K-Fold 모델 병합 코드
    │   ├── pdf_converter.py              # PDF를 이미지로 변환하는 코드
    │   ├── rename_files.py               # 파일명 정리하는 유틸리티
    │   ├── score.py                      # 모델 평가 점수 계산 코드
    │   ├── seeds.py                      # 랜덤 시드 고정 코드
    │── validation_visualization/         # 검증 과정에서 시각화된 이미지 저장 디렉터리
    │── web_service/                      # 웹 서비스 관련 코드 (Django 기반)
    │── .gitignore                        # Git에서 제외할 파일 정의
    │── inference_ensemble.py             # 앙상블 추론 코드
    │── inference.py                      # 단일 모델 추론 코드
    │── loss.py                           # 손실 함수 정의
    │── main.py                           # 메인 실행 스크립트 (학습 및 평가)
    │── README.md                         # 프로젝트 개요 및 사용법 설명
    │── requirements.txt                   # 필요한 Python 패키지 목록
    │── train.py                          # 모델 학습 코드
```

#### 실행 방법
##### 가상환경 설정 (선택 사항)
```bash 
    python -m venv .venv
    source .venv/bin/activate
```

##### 필수 라이브러리 설치
```bash
    pip install -r requirements.txt
```

##### pdf 변환을 위한 필수 패키지 설치
```bash
    # Ubuntu
    sudo apt update
    sudo apt install poppler-utils

    # Mac (Homebrew)
    brew install poppler
```
##### 학습/추론
```bash
    python main.py
    python inference.py
```

## 학습 결과
### K-Fold=5 적용
- K-Fold를 적용하였을 때, Teacher 모델은 100%, Student 모델은 약 80% 정확도를 보였다.
![스크린샷 2025-02-03 132759](https://github.com/user-attachments/assets/7c69acac-8ba2-41ca-bb73-351600be5035)
- 손실값이 널뛰어 Early Stop의 patience를 10으로 설정해야 했다.
- Student의 경우, Loss가 떨어지지 않고 학습이 덜 된 상태에서 학습이 종료되었다.
- 성능이 낮은 데다, 실시간 추론에 5개 Fold를 다 돌려야하는 불편함이 있어서 사용하지 않았다.

### 전체 Train
- Validation없이 Train 전체로 학습(temperature:0.3, alpha:0.7)하였을 때 Student 모델이 100%의 정확도를 보였다.
    - Teacher
      ![스크린샷 2025-02-10 041336](https://github.com/user-attachments/assets/56bcbb4d-9d3d-4615-92d7-a903530bf5f6)
    - Student
      ![스크린샷 2025-02-10 041427](https://github.com/user-attachments/assets/caace1c9-5403-409a-908d-dcd1a4f2b928)

### 기타
- 학습 데이터가 극단적으로 없었음에도, 패턴이 정해져 특징이 매우 뚜렷한 레이아웃을 가졌기 때문에 약간의 기대를 갖고 시작하였다.
- 학습도 얼마 걸리지 않았는데, 성능이 말도 안되게 좋았고 Student도 완벽한 성능을 보였다.
- 생활기록부대체양식의 경우 test data에 포함된 양식이 아니라, 다른 학교의 생활기록부대체양식만 학습에 포함시켰을 때도 정확도가 100%, Confidence가 1.00이었다.
  - 단, 변인이 아니었던 기초생활수급자증명서를 alpha가 0.7일 때 전혀 맞추지 못했고, alpha가 0.8일 때 전부 맞췄다.

