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
