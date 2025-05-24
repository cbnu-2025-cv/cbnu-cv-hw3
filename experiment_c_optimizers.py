"""
실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)

목표: SGD, SGD+Momentum, Adam의 성능 비교
- 학습률 변화가 미치는 영향 분석 (0.1, 0.01, 0.001)
- 학습률 스케줄러(Exponential Decay) 적용 효과 분석
- Overshooting, 느린 수렴, 안정성 확인
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
            nn.Linear(input_size, 256),  # 입력 → 첫 번째 은닉층
            nn.ReLU(),  # 활성화 함수 (ReLU)
            nn.Linear(256, 128),  # 두 번째 은닉층
            nn.ReLU(),  # 활성화 함수 (ReLU)
            nn.Linear(128, num_classes),  # 출력층
        )

    def forward(self, x):
        return self.model(x)


def load_fashion_mnist(use_gpu=True):
    """Fashion-MNIST 데이터셋 로드 (GPU 최적화)"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # GPU 사용 시 최적화된 DataLoader 설정
    num_workers = 4 if use_gpu else 0
    pin_memory = use_gpu
    batch_size = 128 if use_gpu else 64  # GPU 사용 시 배치 크기 증가

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    # GPU 사용 시 최적화된 DataLoader 설정
    num_workers = 2 if use_gpu else 0
    pin_memory = use_gpu
    batch_size = 64 if use_gpu else 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, X.shape[1], 10


def create_optimizer(model_params, optimizer_name, learning_rate):
    """최적화 알고리즘 생성"""
    if optimizer_name == "SGD":
        return optim.SGD(model_params, lr=learning_rate)
    elif optimizer_name == "SGD+Momentum":
        return optim.SGD(model_params, lr=learning_rate, momentum=0.9)
    elif optimizer_name == "Adam":
        return optim.Adam(model_params, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_model_with_scheduler(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    epochs,
    device,
    optimizer_name,
    lr,
    use_scheduler=True,
    use_amp=True,
):
    """학습률 스케줄러를 포함한 모델 훈련 (Mixed Precision 지원)"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Mixed Precision 설정
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # 기록용 리스트들
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    learning_rates = []  # 학습률 변화 기록
    gradient_norms = []  # 그래디언트 노름 기록

    for epoch in tqdm(range(epochs), desc=f"Training {optimizer_name} (lr={lr})"):
        # 훈련 모드
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_gradient_norms = []

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
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                # 그래디언트 노름 계산 (스케일 고려)
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                epoch_gradient_norms.append(total_norm / scaler.get_scale())

                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # 그래디언트 노름 계산
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1.0 / 2)
                epoch_gradient_norms.append(total_norm)

                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 학습률 스케줄러 적용
        if use_scheduler and scheduler is not None:
            scheduler.step()

        # 에포크별 기록
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 현재 학습률 기록
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        # 평균 그래디언트 노름 기록
        avg_grad_norm = np.mean(epoch_gradient_norms) if epoch_gradient_norms else 0
        gradient_norms.append(avg_grad_norm)

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
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 20 에포크마다 출력 및 GPU 메모리 정보
        if (epoch + 1) % 20 == 0:
            memory_info = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                memory_info = f", GPU Mem: {allocated:.1f}/{reserved:.1f} GB"

            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, "
                f"Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}{memory_info}"
            )

        # GPU 메모리 정리 (매 50 에포크마다)
        if epoch % 50 == 0 and device.type == "cuda":
            clear_gpu_memory()

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "learning_rates": learning_rates,
        "gradient_norms": gradient_norms,
    }


def analyze_convergence_stability(losses, window_size=10):
    """수렴 안정성 분석"""
    if len(losses) < window_size:
        return {"volatility": 0, "final_trend": 0}

    # 최근 window_size 구간의 변동성 계산
    recent_losses = losses[-window_size:]
    volatility = np.std(recent_losses)

    # 최근 구간의 트렌드 (기울기)
    x = np.arange(len(recent_losses))
    trend = np.polyfit(x, recent_losses, 1)[0]  # 1차 다항식의 기울기

    return {"volatility": volatility, "final_trend": trend}


def plot_comprehensive_results(results_dict, dataset_name, learning_rate):
    """포괄적인 결과 시각화"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle(
        f"실험 C: 최적화 알고리즘 비교 ({dataset_name}, LR={learning_rate})",
        fontsize=16,
    )

    # 1. 훈련 손실
    axes[0, 0].set_title("Training Loss")
    for opt_name, results in results_dict.items():
        axes[0, 0].plot(results["train_losses"], label=opt_name)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_yscale("log")  # 로그 스케일로 변화 패턴 강조

    # 2. 테스트 정확도
    axes[0, 1].set_title("Test Accuracy")
    for opt_name, results in results_dict.items():
        axes[0, 1].plot(results["test_accuracies"], label=opt_name)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. 학습률 변화
    axes[1, 0].set_title("Learning Rate Schedule")
    for opt_name, results in results_dict.items():
        axes[1, 0].plot(results["learning_rates"], label=opt_name)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Learning Rate")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale("log")

    # 4. 그래디언트 노름
    axes[1, 1].set_title("Gradient Norm")
    for opt_name, results in results_dict.items():
        axes[1, 1].plot(results["gradient_norms"], label=opt_name, alpha=0.7)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Gradient Norm")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale("log")

    # 5. 손실 스무딩 (이동평균)
    axes[2, 0].set_title("Smoothed Training Loss (Moving Average)")
    window_size = 5
    for opt_name, results in results_dict.items():
        losses = results["train_losses"]
        if len(losses) >= window_size:
            smoothed = pd.Series(losses).rolling(window=window_size).mean()
            axes[2, 0].plot(smoothed, label=opt_name)
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Smoothed Loss")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # 6. 수렴 분석 (최근 구간 확대)
    axes[2, 1].set_title("Convergence Analysis (Last 20 Epochs)")
    for opt_name, results in results_dict.items():
        recent_losses = results["test_losses"][-20:]
        axes[2, 1].plot(
            range(len(results["test_losses"]) - 20, len(results["test_losses"])),
            recent_losses,
            label=opt_name,
            marker="o",
            markersize=3,
        )
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Test Loss")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    filename = f'experiment_c_results_{dataset_name.lower().replace("-", "_")}_lr_{str(learning_rate).replace(".", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return filename


def create_detailed_comparison_table(all_results, learning_rates):
    """학습률별 상세 비교 표 생성"""
    comparison_data = []

    for lr in learning_rates:
        for dataset_name, dataset_results in all_results.items():
            if lr not in dataset_results:
                continue

            for opt_name, results in dataset_results[lr].items():
                final_train_acc = results["train_accuracies"][-1]
                final_test_acc = results["test_accuracies"][-1]
                min_train_loss = min(results["train_losses"])
                best_test_acc = max(results["test_accuracies"])

                # 수렴 속도 (최고 정확도의 95%에 도달한 시점)
                convergence_threshold = best_test_acc * 0.95
                convergence_epoch = next(
                    (
                        i
                        for i, acc in enumerate(results["test_accuracies"])
                        if acc >= convergence_threshold
                    ),
                    len(results["test_accuracies"]),
                )

                # 안정성 분석
                stability = analyze_convergence_stability(results["test_losses"])

                # 평균 그래디언트 노름
                avg_grad_norm = np.mean(results["gradient_norms"])

                comparison_data.append(
                    {
                        "데이터셋": dataset_name,
                        "최적화기": opt_name,
                        "학습률": lr,
                        "최종 테스트 정확도 (%)": f"{final_test_acc:.2f}",
                        "최고 테스트 정확도 (%)": f"{best_test_acc:.2f}",
                        "최소 훈련 손실": f"{min_train_loss:.4f}",
                        "수렴 속도 (epochs)": convergence_epoch + 1,
                        "안정성 (변동성)": f"{stability['volatility']:.4f}",
                        "평균 Gradient Norm": f"{avg_grad_norm:.4f}",
                    }
                )

    df = pd.DataFrame(comparison_data)
    return df


def plot_learning_rate_comparison(all_results, dataset_name, optimizer_name):
    """학습률별 비교 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{optimizer_name} - 학습률별 비교 ({dataset_name})", fontsize=16)

    colors = ["blue", "red", "green"]

    for idx, (lr, color) in enumerate(zip([0.1, 0.01, 0.001], colors)):
        if lr not in all_results[dataset_name]:
            continue
        if optimizer_name not in all_results[dataset_name][lr]:
            continue

        results = all_results[dataset_name][lr][optimizer_name]

        # 훈련 손실
        axes[0, 0].plot(results["train_losses"], label=f"LR={lr}", color=color)
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale("log")

        # 테스트 정확도
        axes[0, 1].plot(results["test_accuracies"], label=f"LR={lr}", color=color)
        axes[0, 1].set_title("Test Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 학습률 변화
        axes[1, 0].plot(results["learning_rates"], label=f"LR={lr}", color=color)
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale("log")

        # 그래디언트 노름
        axes[1, 1].plot(
            results["gradient_norms"], label=f"LR={lr}", color=color, alpha=0.7
        )
        axes[1, 1].set_title("Gradient Norm")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Gradient Norm")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale("log")

    plt.tight_layout()
    filename = (
        f'lr_comparison_{optimizer_name}_{dataset_name.lower().replace("-", "_")}.png'
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    return filename


def run_experiment_c():
    """실험 C 메인 실행 함수"""
    print("=" * 60)
    print("실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)")
    print("=" * 60)

    # GPU 설정
    device, use_gpu = setup_device()
    use_amp = use_gpu  # Mixed Precision은 GPU에서만 사용

    # 실험 설정
    epochs = 100
    learning_rates = [0.1, 0.01, 0.001]
    optimizers = ["SGD", "SGD+Momentum", "Adam"]
    use_scheduler = True  # Exponential Decay 사용 여부

    # 데이터셋별 실험
    datasets = [("Fashion-MNIST", load_fashion_mnist), ("Digits", load_digits_dataset)]

    all_results = {}

    for dataset_name, load_func in datasets:
        print(f"\n{'='*50}")
        print(f"--- {dataset_name} 데이터셋으로 실험 ---")
        print(f"{'='*50}")

        # 데이터 로드
        train_loader, test_loader, input_size, num_classes = load_func(use_gpu)

        all_results[dataset_name] = {}

        for lr in learning_rates:
            print(f"\n--- 학습률 {lr} ---")
            all_results[dataset_name][lr] = {}

            results_for_lr = {}

            for optimizer_name in optimizers:
                print(f"\n{optimizer_name} (LR={lr}) 최적화기로 학습 중...")

                # GPU 메모리 정리
                clear_gpu_memory()

                # 모델 초기화 (매번 새로 생성하여 공정한 비교)
                model = MLP(input_size, num_classes)

                # 최적화기 생성
                optimizer = create_optimizer(model.parameters(), optimizer_name, lr)

                # 스케줄러 생성 (Exponential Decay)
                scheduler = None
                if use_scheduler:
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

                # 모델 훈련
                results = train_model_with_scheduler(
                    model,
                    train_loader,
                    test_loader,
                    optimizer,
                    scheduler,
                    epochs,
                    device,
                    optimizer_name,
                    lr,
                    use_scheduler,
                    use_amp,
                )

                results_for_lr[optimizer_name] = results
                all_results[dataset_name][lr][optimizer_name] = results

                print(f"최종 테스트 정확도: {results['test_accuracies'][-1]:.2f}%")
                print(f"최고 테스트 정확도: {max(results['test_accuracies']):.2f}%")

                # 모델 삭제 및 GPU 메모리 정리
                del model, optimizer
                if scheduler:
                    del scheduler
                clear_gpu_memory()

            # 학습률별 결과 시각화
            plot_filename = plot_comprehensive_results(results_for_lr, dataset_name, lr)
            print(f"결과 그래프 저장: {plot_filename}")

        # 최적화기별 학습률 비교
        for optimizer_name in optimizers:
            lr_comp_filename = plot_learning_rate_comparison(
                all_results, dataset_name, optimizer_name
            )
            print(f"학습률 비교 그래프 저장: {lr_comp_filename}")

    # 전체 결과 정량적 비교
    print("\n" + "=" * 60)
    print("전체 실험 결과 정량적 비교")
    print("=" * 60)

    comparison_df = create_detailed_comparison_table(all_results, learning_rates)
    print(comparison_df.to_string(index=False))

    # 결과를 CSV로 저장
    comparison_df.to_csv(
        "experiment_c_comprehensive_comparison.csv", index=False, encoding="utf-8-sig"
    )

    # 최고 성능 분석
    print("\n" + "=" * 40)
    print("최고 성능 분석")
    print("=" * 40)

    for dataset_name in all_results.keys():
        print(f"\n{dataset_name} 데이터셋:")
        best_accuracy = 0
        best_config = None

        for lr in learning_rates:
            for opt_name in optimizers:
                if (
                    lr in all_results[dataset_name]
                    and opt_name in all_results[dataset_name][lr]
                ):
                    max_acc = max(
                        all_results[dataset_name][lr][opt_name]["test_accuracies"]
                    )
                    if max_acc > best_accuracy:
                        best_accuracy = max_acc
                        best_config = (opt_name, lr)

        if best_config:
            print(
                f"  최고 성능: {best_config[0]} (LR={best_config[1]}) - {best_accuracy:.2f}%"
            )


if __name__ == "__main__":
    # 실험 C 실행
    run_experiment_c()

    print("\n" + "=" * 60)
    print("실험 C 완료!")
    print("결과 파일들:")
    print("- experiment_c_results_*.png (학습률별 상세 결과)")
    print("- lr_comparison_*.png (최적화기별 학습률 비교)")
    print("- experiment_c_comprehensive_comparison.csv (전체 정량적 비교)")
    print("=" * 60)
