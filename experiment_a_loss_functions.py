"""
실험 A: 손실 함수 비교 (CrossEntropy Loss vs MSE Loss with softmax)

목표: MSE와 CrossEntropy가 학습 성능에 미치는 차이 분석
- 학습 곡선의 수렴 속도, 정확도, loss 안정성 비교
- MSE 사용 시 Gradient Vanishing 문제 분석
- CrossEntropy의 빠른 수렴 속도와 안정성 확인
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore")

# 시각화 설정
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_style("whitegrid")


# GPU 설정 최적화
def setup_device():
    """GPU 설정 및 최적화"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # GPU 정보 출력
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)

        print(f"🚀 GPU 사용 가능!")
        print(f"📊 GPU 개수: {gpu_count}")
        print(f"🎯 현재 GPU: {current_gpu} ({gpu_name})")

        # GPU 메모리 정보
        total_memory = torch.cuda.get_device_properties(current_gpu).total_memory / (
            1024**3
        )
        print(f"💾 GPU 메모리: {total_memory:.1f} GB")

        return device, True
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU를 사용합니다.")
        return torch.device("cpu"), False


def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class MLP(nn.Module):
    """기본 MLP 네트워크 구조"""

    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),  # 입력 → 첫 번째 은닉층 (input_size → 256)
            nn.ReLU(),  # 활성화 함수 (ReLU)
            nn.Linear(256, 128),  # 두 번째 은닉층 (256 → 128)
            nn.ReLU(),  # 활성화 함수 (ReLU)
            nn.Linear(128, num_classes),  # 출력층 (128 → 클래스 개수)
        )

    def forward(self, x):
        return self.model(x)


def load_fashion_mnist(use_gpu=True):
    """Fashion-MNIST 데이터셋 로드 (GPU 최적화)"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]  # 정규화
    )

    # 훈련 데이터셋
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # 테스트 데이터셋
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # GPU 사용 시 최적화된 DataLoader 설정
    num_workers = 4 if use_gpu else 0
    pin_memory = use_gpu

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,  # GPU 사용 시 배치 크기 증가
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, test_loader, 28 * 28, 10


def load_digits_dataset(use_gpu=True):
    """Scikit-learn Digits 데이터셋 로드 (GPU 최적화)"""
    digits = load_digits()
    X, y = digits.data, digits.target

    # 데이터 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # PyTorch 텐서로 변환
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # DataLoader 생성
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # GPU 사용 시 최적화된 DataLoader 설정
    num_workers = 2 if use_gpu else 0
    pin_memory = use_gpu

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, X.shape[1], 10


def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs,
    device,
    loss_name,
    use_amp=True,
):
    """모델 훈련 함수 (Mixed Precision 지원)"""
    model.to(device)

    # Mixed Precision 설정
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # 기록용 리스트들
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in tqdm(range(epochs), desc=f"Training with {loss_name}"):
        # 훈련 모드
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )

            # 입력 데이터를 1차원으로 변환 (Fashion-MNIST의 경우)
            if len(inputs.shape) > 2:
                inputs = inputs.view(inputs.size(0), -1)

            optimizer.zero_grad(set_to_none=True)  # 메모리 효율성 향상

            # Mixed Precision Forward Pass
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)

                    # MSE Loss 사용 시 softmax 적용
                    if isinstance(loss_fn, nn.MSELoss):
                        # One-hot encoding for MSE
                        labels_onehot = torch.zeros(labels.size(0), 10, device=device)
                        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

                        # Softmax 적용
                        outputs = torch.softmax(outputs, dim=1)
                        loss = loss_fn(outputs, labels_onehot)

                        # 정확도 계산을 위한 예측
                        _, predicted = torch.max(outputs.data, 1)
                    else:
                        # CrossEntropy Loss
                        loss = loss_fn(outputs, labels)
                        _, predicted = torch.max(outputs.data, 1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)

                # MSE Loss 사용 시 softmax 적용
                if isinstance(loss_fn, nn.MSELoss):
                    # One-hot encoding for MSE
                    labels_onehot = torch.zeros(labels.size(0), 10, device=device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)

                    # Softmax 적용
                    outputs = torch.softmax(outputs, dim=1)
                    loss = loss_fn(outputs, labels_onehot)

                    # 정확도 계산을 위한 예측
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    # CrossEntropy Loss
                    loss = loss_fn(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)

                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 에포크별 훈련 손실과 정확도
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 테스트 평가
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )

                if len(inputs.shape) > 2:
                    inputs = inputs.view(inputs.size(0), -1)

                # Mixed Precision Inference
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if isinstance(loss_fn, nn.MSELoss):
                    labels_onehot = torch.zeros(labels.size(0), 10, device=device)
                    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                    outputs = torch.softmax(outputs, dim=1)
                    loss = loss_fn(outputs, labels_onehot)
                    _, predicted = torch.max(outputs.data, 1)
                else:
                    loss = loss_fn(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)

                test_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 10 에포크마다 출력 및 GPU 메모리 정보
        if (epoch + 1) % 10 == 0:
            memory_info = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                memory_info = f", GPU Mem: {allocated:.1f}/{reserved:.1f} GB"

            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%{memory_info}"
            )

    return train_losses, train_accuracies, test_losses, test_accuracies


def analyze_gradient_flow(model, train_loader, loss_fn, device):
    """그래디언트 흐름 분석"""
    model.eval()
    gradients = []

    inputs, labels = next(iter(train_loader))
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(
        device, non_blocking=True
    )

    if len(inputs.shape) > 2:
        inputs = inputs.view(inputs.size(0), -1)

    outputs = model(inputs)

    if isinstance(loss_fn, nn.MSELoss):
        labels_onehot = torch.zeros(labels.size(0), 10, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        outputs = torch.softmax(outputs, dim=1)
        loss = loss_fn(outputs, labels_onehot)
    else:
        loss = loss_fn(outputs, labels)

    loss.backward()

    # 각 레이어의 그래디언트 수집
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append((name, param.grad.abs().mean().item()))

    return gradients


def plot_results(results_dict, dataset_name):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"실험 A: 손실 함수 비교 결과 ({dataset_name})", fontsize=16)

    # 훈련 손실
    axes[0, 0].set_title("Training Loss")
    for loss_name, metrics in results_dict.items():
        axes[0, 0].plot(metrics["train_losses"], label=loss_name)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 테스트 손실
    axes[0, 1].set_title("Test Loss")
    for loss_name, metrics in results_dict.items():
        axes[0, 1].plot(metrics["test_losses"], label=loss_name)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 훈련 정확도
    axes[1, 0].set_title("Training Accuracy")
    for loss_name, metrics in results_dict.items():
        axes[1, 0].plot(metrics["train_accuracies"], label=loss_name)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 테스트 정확도
    axes[1, 1].set_title("Test Accuracy")
    for loss_name, metrics in results_dict.items():
        axes[1, 1].plot(metrics["test_accuracies"], label=loss_name)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        f'experiment_a_results_{dataset_name.lower().replace("-", "_")}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def create_comparison_table(results_dict):
    """정량적 비교 표 생성"""
    comparison_data = []

    for loss_name, metrics in results_dict.items():
        final_train_acc = metrics["train_accuracies"][-1]
        final_test_acc = metrics["test_accuracies"][-1]
        min_train_loss = min(metrics["train_losses"])
        min_test_loss = min(metrics["test_losses"])

        # 수렴까지 걸린 에포크 수 (테스트 정확도가 최고점의 95%에 도달한 시점)
        max_acc = max(metrics["test_accuracies"])
        convergence_threshold = max_acc * 0.95
        convergence_epoch = next(
            (
                i
                for i, acc in enumerate(metrics["test_accuracies"])
                if acc >= convergence_threshold
            ),
            len(metrics["test_accuracies"]),
        )

        comparison_data.append(
            {
                "손실 함수": loss_name,
                "최종 훈련 정확도 (%)": f"{final_train_acc:.2f}",
                "최종 테스트 정확도 (%)": f"{final_test_acc:.2f}",
                "최소 훈련 손실": f"{min_train_loss:.4f}",
                "최소 테스트 손실": f"{min_test_loss:.4f}",
                "수렴까지 걸린 에포크": convergence_epoch + 1,
            }
        )

    df = pd.DataFrame(comparison_data)
    print("\n=== 정량적 비교 결과 ===")
    print(df.to_string(index=False))

    return df


def run_experiment_a():
    """실험 A 메인 실행 함수"""
    print("=" * 60)
    print("실험 A: 손실 함수 비교 (CrossEntropy vs MSE with softmax)")
    print("=" * 60)

    # GPU 설정
    device, use_gpu = setup_device()
    use_amp = use_gpu  # Mixed Precision은 GPU에서만 사용

    # 실험 설정
    epochs = 30
    learning_rate = 0.001 if use_gpu else 0.001

    # 데이터셋별 실험
    datasets = [("Fashion-MNIST", load_fashion_mnist), ("Digits", load_digits_dataset)]

    for dataset_name, load_func in datasets:
        print(f"\n--- {dataset_name} 데이터셋으로 실험 ---")

        # GPU 메모리 정리
        clear_gpu_memory()

        # 데이터 로드
        train_loader, test_loader, input_size, num_classes = load_func(use_gpu)

        # 손실 함수들
        loss_functions = {
            "CrossEntropy": nn.CrossEntropyLoss(),
            "MSE (with softmax)": nn.MSELoss(),
        }

        results = {}

        for loss_name, loss_fn in loss_functions.items():
            print(f"\n{loss_name} 손실 함수로 학습 중...")

            # 모델 초기화 (매번 새로 생성하여 공정한 비교)
            model = MLP(input_size, num_classes)

            # GPU 사용 시 AdamW 옵티마이저 사용 (더 안정적)
            if use_gpu:
                optimizer = optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=0.01
                )
            else:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 모델 훈련
            train_losses, train_accs, test_losses, test_accs = train_model(
                model,
                train_loader,
                test_loader,
                loss_fn,
                optimizer,
                epochs,
                device,
                loss_name,
                use_amp,
            )

            results[loss_name] = {
                "train_losses": train_losses,
                "train_accuracies": train_accs,
                "test_losses": test_losses,
                "test_accuracies": test_accs,
                "model": model,
            }

            # 그래디언트 흐름 분석
            print(f"\n{loss_name} 그래디언트 흐름 분석:")
            gradients = analyze_gradient_flow(model, train_loader, loss_fn, device)
            for name, grad_val in gradients:
                print(f"  {name}: {grad_val:.6f}")

            # GPU 메모리 정리
            del model, optimizer
            clear_gpu_memory()

        # 결과 시각화
        plot_results(results, dataset_name)

        # 정량적 비교
        comparison_df = create_comparison_table(results)

        # 결과를 CSV로 저장
        comparison_df.to_csv(
            f'experiment_a_comparison_{dataset_name.lower().replace("-", "_")}.csv',
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    # 실험 A 실행
    run_experiment_a()

    print("\n" + "=" * 60)
    print("실험 A 완료!")
    print("결과 파일들:")
    print("- experiment_a_results_fashion_mnist.png")
    print("- experiment_a_results_digits.png")
    print("- experiment_a_comparison_fashion_mnist.csv")
    print("- experiment_a_comparison_digits.csv")
    print("=" * 60)
