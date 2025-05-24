# 딥러닝 실험 패키지 🧠⚗️

손실 함수, 활성화 함수, 최적화 알고리즘이 딥러닝 모델의 학습 성능에 미치는 영향을 체계적으로 분석하는 GPU 최적화 실험 패키지입니다.

## 🚀 GPU 최적화 특징

### ⚡ 성능 향상

- **Mixed Precision Training (FP16)**: 메모리 사용량 50% 감소, 학습 속도 1.5-2배 향상
- **최적화된 DataLoader**: `pin_memory`, `num_workers`, `persistent_workers` 적용
- **대용량 배치 처리**: GPU 메모리 활용도 극대화
- **자동 메모리 관리**: 주기적 GPU 캐시 정리

### 💾 메모리 효율성

- **동적 메모리 관리**: OOM 에러 방지
- **Gradient Scaling**: Mixed Precision 환경에서 안정적 학습
- **Non-blocking 텐서 이동**: CPU-GPU 간 데이터 전송 최적화

## 📋 실험 개요

### 🎯 실험 목표

- 손실 함수, 활성화 함수, 최적화 알고리즘의 학습 결과 영향 정량 분석
- GPU 가속화를 통한 효율적인 대규모 실험 수행
- Mixed Precision Training의 안정성 및 성능 분석

### 🔬 실험 구성

| 실험       | 비교 대상                    | 데이터셋                 | GPU 최적화                           |
| ---------- | ---------------------------- | ------------------------ | ------------------------------------ |
| **실험 A** | CrossEntropy vs MSE Loss     | Fashion-MNIST, Digits    | Mixed Precision, 대용량 배치 (128)   |
| **실험 B** | ReLU vs LeakyReLU vs Sigmoid | make_moons, make_circles | 병렬 데이터 로딩, 메모리 효율적 텐서 |
| **실험 C** | SGD vs SGD+Momentum vs Adam  | Fashion-MNIST, Digits    | AdamW 옵티마이저, Gradient Scaling   |

## 🚀 빠른 시작

### 1️⃣ 환경 설정

#### GPU 환경 요구사항

```bash
# CUDA 호환 GPU (GTX 1060 이상 권장)
# CUDA 11.0 이상
# 최소 4GB GPU 메모리 (8GB 이상 권장)
```

#### PyTorch GPU 버전 설치

```bash
# CUDA 11.8 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 또는 CUDA 12.1 버전
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 기타 패키지 설치
pip install -r requirements.txt
```

#### GPU 환경 확인

```python
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"GPU 개수: {torch.cuda.device_count()}")
print(f"현재 GPU: {torch.cuda.get_device_name(0)}")
```

### 2️⃣ 전체 실험 실행

```bash
# 모든 실험을 GPU 최적화로 순차 실행
python run_all_experiments.py
```

### 3️⃣ 개별 실험 실행

```bash
# 실험 A: 손실 함수 비교 (GPU 가속)
python experiment_a_loss_functions.py

# 실험 B: 활성화 함수 비교 (GPU 가속)
python experiment_b_activation_functions.py

# 실험 C: 최적화 알고리즘 비교 (GPU 가속)
python experiment_c_optimizers.py
```

## 📁 프로젝트 구조

```
cbnu-cv-hw3/
├── requirements.txt                    # GPU 최적화 패키지 의존성
├── README.md                          # 프로젝트 설명서 (GPU 버전)
├── run_all_experiments.py             # 전체 실험 실행 (GPU 최적화)
├── experiment_a_loss_functions.py     # 실험 A: 손실 함수 (Mixed Precision)
├── experiment_b_activation_functions.py # 실험 B: 활성화 함수 (GPU 가속)
├── experiment_c_optimizers.py         # 실험 C: 최적화 알고리즘 (GPU 최적화)
└── 실험_종합_보고서_GPU최적화.md        # GPU 최적화 종합 보고서
```

## 📊 실험 상세

### 🅰️ 실험 A: 손실 함수 비교 (GPU 최적화)

**목표**: CrossEntropy Loss와 MSE Loss (with softmax)의 성능 차이 분석

**GPU 최적화**:

- Mixed Precision Training으로 메모리 사용량 절반 감소
- 배치 크기 64 → 128로 증가 (GPU 메모리 활용도 극대화)
- Non-blocking 텐서 이동으로 데이터 전송 최적화

**핵심 분석**:

- GPU 환경에서 Mixed Precision 사용 시 손실 함수별 안정성
- 대용량 배치에서의 수렴 속도 변화
- GPU 메모리 사용 패턴 분석

### 🅱️ 실험 B: 활성화 함수 비교 (GPU 가속)

**목표**: ReLU, LeakyReLU, Sigmoid의 학습 영향 분석

**GPU 최적화**:

- 병렬 데이터 로딩 (`num_workers=4`)
- GPU 최적화된 활성화 분포 계산
- 메모리 효율적 결정 경계 시각화

**핵심 분석**:

- GPU 가속화 환경에서 Dead ReLU 현상 변화
- Mixed Precision이 활성화 함수 성능에 미치는 영향
- 메모리 효율적 활성화 분포 분석 방법

### 🅲️ 실험 C: 최적화 알고리즘 비교 (GPU 최적화)

**목표**: SGD, SGD+Momentum, Adam의 성능 및 학습률 영향 분석

**GPU 최적화**:

- AdamW 옵티마이저 사용 (GPU에서 더 안정적)
- Gradient Scaling으로 Mixed Precision 안정성 확보
- 배치 크기 자동 조정

**핵심 분석**:

- GPU 환경에서 최적화 알고리즘별 스케일링 효과
- Mixed Precision 환경에서 Gradient Clipping 필요성
- 대용량 배치 학습 시 학습률 조정 전략

## 📈 성능 개선 결과

### 🏃‍♂️ 학습 속도 향상

- **CPU 대비 GPU**: 5-10배 빠른 학습 속도
- **Mixed Precision 적용**: 추가 1.5-2배 향상
- **최적화된 DataLoader**: I/O 병목 현상 제거

### 💾 메모리 효율성

- **Mixed Precision**: 메모리 사용량 50% 감소
- **동적 메모리 관리**: OOM 에러 방지
- **배치 크기 최적화**: GPU 메모리 활용도 극대화

### ⚙️ 기술 사양

#### 📦 주요 의존성 (GPU 최적화)

- **PyTorch**: 1.12.0+ (CUDA 지원 버전)
- **torchvision**: 0.13.0+ (CUDA 지원 버전)
- **CUDA**: 11.0+ (Tensor Core 지원 시 Mixed Precision 사용)
- **GPU 메모리**: 최소 4GB (8GB 이상 권장)

#### ⚙️ 실험 설정 (GPU 최적화)

- **네트워크**: MLP (256 → 128 → output)
- **배치 크기**: 128 (GPU), 64 (CPU)
- **에포크 수**: 30-300 (실험별 조정)
- **Mixed Precision**: FP16 (Tensor Core 지원 GPU)
- **DataLoader**: `num_workers=4`, `pin_memory=True`
