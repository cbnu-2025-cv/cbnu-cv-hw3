"""
실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)

목표: ReLU, LeakyReLU, Sigmoid가 학습에 미치는 영향을 분석
- Dead ReLU 발생 유도 및 LeakyReLU의 완화 효과 확인
- Sigmoid의 vanishing gradient 문제 분석
- Layer별 출력값 및 뉴런 활성화 패턴 시각화
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
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


class MLPWithActivation(nn.Module):
    """활성화 함수를 변경 가능한 MLP 네트워크"""

    def __init__(self, input_size, num_classes, activation_fn, small_weights=True):
        super(MLPWithActivation, self).__init__()

        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, num_classes)

        # 활성화 함수 설정
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation_fn == "sigmoid":
            self.activation = nn.Sigmoid()

        # Dead ReLU 유도를 위한 작은 가중치 초기화
        if small_weights:
            self._initialize_small_weights()

    def _initialize_small_weights(self):
        """작은 가중치로 초기화하여 Dead ReLU 상황 유도"""
        for layer in [self.layer1, self.layer2, self.layer3]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 중간 레이어 출력값을 저장하기 위한 리스트
        activations = []

        x = self.layer1(x)
        x = self.activation(x)
        activations.append(x.clone())

        x = self.layer2(x)
        x = self.activation(x)
        activations.append(x.clone())

        x = self.layer3(x)

        return x, activations


def load_2d_datasets(use_gpu=True):
    """2D 비선형 분류 데이터셋 생성 (GPU 최적화)"""
    datasets = {}

    # make_moons 데이터셋
    X_moons, y_moons = make_moons(n_samples=1000, noise=0.3, random_state=42)
    scaler = StandardScaler()
    X_moons = scaler.fit_transform(X_moons)

    X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
        X_moons, y_moons, test_size=0.3, random_state=42, stratify=y_moons
    )

    # GPU에서 사용할 경우 텐서를 미리 GPU로 이동
    device_type = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    datasets["moons"] = {
        "X_train": torch.FloatTensor(X_train_moons),
        "X_test": torch.FloatTensor(X_test_moons),
        "y_train": torch.LongTensor(y_train_moons),
        "y_test": torch.LongTensor(y_test_moons),
        "input_size": 2,
        "num_classes": 2,
    }

    # make_circles 데이터셋
    X_circles, y_circles = make_circles(
        n_samples=1000, noise=0.2, factor=0.5, random_state=42
    )
    scaler = StandardScaler()
    X_circles = scaler.fit_transform(X_circles)

    X_train_circles, X_test_circles, y_train_circles, y_test_circles = train_test_split(
        X_circles, y_circles, test_size=0.3, random_state=42, stratify=y_circles
    )

    datasets["circles"] = {
        "X_train": torch.FloatTensor(X_train_circles),
        "X_test": torch.FloatTensor(X_test_circles),
        "y_train": torch.LongTensor(y_train_circles),
        "y_test": torch.LongTensor(y_test_circles),
        "input_size": 2,
        "num_classes": 2,
    }

    return datasets


def calculate_dead_relu_ratio(activations):
    """Dead ReLU 비율 계산"""
    dead_ratios = []

    for layer_activation in activations:
        # 0 이하의 값을 가진 뉴런 비율 계산
        total_neurons = layer_activation.numel()
        dead_neurons = (layer_activation <= 0).sum().item()
        dead_ratio = dead_neurons / total_neurons * 100
        dead_ratios.append(dead_ratio)

    return dead_ratios


def train_model_with_analysis(
    model, data, epochs, learning_rate, device, activation_name, use_amp=True
):
    """활성화 함수 분석을 포함한 모델 훈련 (Mixed Precision 지원)"""
    model.to(device)

    X_train, X_test = data["X_train"].to(device, non_blocking=True), data["X_test"].to(
        device, non_blocking=True
    )
    y_train, y_test = data["y_train"].to(device, non_blocking=True), data["y_test"].to(
        device, non_blocking=True
    )

    # GPU 사용 시 AdamW 옵티마이저 사용 (더 안정적)
    if device.type == "cuda":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # Mixed Precision 설정
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    # 기록용 리스트들
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    dead_relu_history = []  # Dead ReLU 비율 기록
    activation_distributions = []  # 활성화 분포 기록

    for epoch in tqdm(range(epochs), desc=f"Training with {activation_name}"):
        # 훈련 모드
        model.train()

        # Mixed Precision Forward Pass
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs, activations = model(X_train)
                loss = criterion(outputs, y_train)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, activations = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # 훈련 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        train_acc = (predicted == y_train).sum().item() / y_train.size(0) * 100

        train_losses.append(loss.item())
        train_accuracies.append(train_acc)

        # 테스트 평가
        model.eval()
        with torch.no_grad():
            # Mixed Precision Inference
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    test_outputs, test_activations = model(X_test)
                    test_loss = criterion(test_outputs, y_test)
            else:
                test_outputs, test_activations = model(X_test)
                test_loss = criterion(test_outputs, y_test)

            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == y_test).sum().item() / y_test.size(0) * 100

        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc)

        # Dead ReLU 분석 (ReLU 계열 활성화 함수에 대해서만)
        if activation_name in ["ReLU", "LeakyReLU"]:
            dead_ratios = calculate_dead_relu_ratio(activations)
            dead_relu_history.append(dead_ratios)

        # 활성화 분포 저장 (특정 에포크에만) - 메모리 절약을 위해 CPU로 이동
        if epoch % (epochs // 5) == 0:  # 5개의 시점에서 기록
            activation_dist = []
            for layer_act in activations:
                activation_dist.append(layer_act.detach().cpu().numpy().flatten())
            activation_distributions.append(
                {"epoch": epoch, "distributions": activation_dist}
            )

        # GPU 메모리 정리 (매 50 에포크마다)
        if epoch % 50 == 0 and device.type == "cuda":
            clear_gpu_memory()

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "dead_relu_history": dead_relu_history,
        "activation_distributions": activation_distributions,
        "model": model,
    }


def analyze_gradients(model, data, device):
    """그래디언트 흐름 분석"""
    model.train()

    X_train = data["X_train"].to(device, non_blocking=True)
    y_train = data["y_train"].to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss()

    outputs, _ = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()

    gradients = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().mean().item()
            gradients.append((name, grad_norm))

    return gradients


def plot_2d_decision_boundary(model, data, dataset_name, activation_name):
    """2D 데이터의 결정 경계 시각화"""
    model.eval()

    X_test = data["X_test"].cpu().numpy()  # CPU로 이동하여 시각화
    y_test = data["y_test"].cpu().numpy()

    # 그리드 생성
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    # GPU에서 추론하고 CPU로 결과 이동
    if next(model.parameters()).is_cuda:
        grid_points = grid_points.cuda()

    with torch.no_grad():
        Z, _ = model(grid_points)
        Z = torch.softmax(Z, dim=1)[:, 1].cpu().numpy()

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors="black"
    )
    plt.colorbar(scatter)
    plt.title(f"Decision Boundary: {activation_name} on {dataset_name}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(
        f"decision_boundary_{activation_name}_{dataset_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_activation_distributions(results_dict, dataset_name):
    """활성화 함수별 뉴런 출력 분포 시각화"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Layer별 활성화 분포 ({dataset_name})", fontsize=16)

    for idx, (activation_name, results) in enumerate(results_dict.items()):
        if not results["activation_distributions"]:
            continue

        # 마지막 에포크의 활성화 분포 사용
        last_dist = results["activation_distributions"][-1]["distributions"]

        # Layer 1
        axes[idx, 0].hist(last_dist[0], bins=50, alpha=0.7, density=True)
        axes[idx, 0].set_title(f"{activation_name} - Layer 1")
        axes[idx, 0].set_xlabel("Activation Value")
        axes[idx, 0].set_ylabel("Density")

        # Layer 2
        axes[idx, 1].hist(last_dist[1], bins=50, alpha=0.7, density=True)
        axes[idx, 1].set_title(f"{activation_name} - Layer 2")
        axes[idx, 1].set_xlabel("Activation Value")
        axes[idx, 1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(
        f"activation_distributions_{dataset_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_dead_relu_heatmap(results_dict, dataset_name):
    """Dead ReLU 비율 히트맵 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Dead ReLU 비율 변화 ({dataset_name})", fontsize=16)

    for idx, (activation_name, results) in enumerate(results_dict.items()):
        if (
            activation_name not in ["ReLU", "LeakyReLU"]
            or not results["dead_relu_history"]
        ):
            continue

        # Dead ReLU 히스토리를 배열로 변환
        dead_relu_array = np.array(results["dead_relu_history"])

        # 히트맵 생성
        im = axes[idx].imshow(dead_relu_array.T, cmap="Reds", aspect="auto")
        axes[idx].set_title(f"{activation_name} Dead Neuron Ratio (%)")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel("Layer")
        axes[idx].set_yticks([0, 1])
        axes[idx].set_yticklabels(["Layer 1", "Layer 2"])

        # 컬러바 추가
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig(f"dead_relu_heatmap_{dataset_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_learning_curves(results_dict, dataset_name):
    """학습 곡선 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"실험 B: 활성화 함수 비교 결과 ({dataset_name})", fontsize=16)

    # 훈련 손실
    axes[0, 0].set_title("Training Loss")
    for activation_name, results in results_dict.items():
        axes[0, 0].plot(results["train_losses"], label=activation_name)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 테스트 손실
    axes[0, 1].set_title("Test Loss")
    for activation_name, results in results_dict.items():
        axes[0, 1].plot(results["test_losses"], label=activation_name)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 훈련 정확도
    axes[1, 0].set_title("Training Accuracy")
    for activation_name, results in results_dict.items():
        axes[1, 0].plot(results["train_accuracies"], label=activation_name)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 테스트 정확도
    axes[1, 1].set_title("Test Accuracy")
    for activation_name, results in results_dict.items():
        axes[1, 1].plot(results["test_accuracies"], label=activation_name)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        f"experiment_b_results_{dataset_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def create_comparison_table(results_dict, gradients_dict):
    """정량적 비교 표 생성"""
    comparison_data = []

    for activation_name, results in results_dict.items():
        final_train_acc = results["train_accuracies"][-1]
        final_test_acc = results["test_accuracies"][-1]

        # Dead ReLU 비율 (ReLU 계열에 대해서만)
        if activation_name in ["ReLU", "LeakyReLU"] and results["dead_relu_history"]:
            avg_dead_ratio = np.mean(
                [np.mean(ratios) for ratios in results["dead_relu_history"]]
            )
        else:
            avg_dead_ratio = 0.0

        # 수렴 속도 (최고 정확도의 95%에 도달한 시점)
        max_acc = max(results["test_accuracies"])
        convergence_threshold = max_acc * 0.95
        convergence_epoch = next(
            (
                i
                for i, acc in enumerate(results["test_accuracies"])
                if acc >= convergence_threshold
            ),
            len(results["test_accuracies"]),
        )

        # 평균 그래디언트 크기
        avg_gradient = np.mean([grad for _, grad in gradients_dict[activation_name]])

        comparison_data.append(
            {
                "활성화 함수": activation_name,
                "Dead ReLU 비율 (%)": f"{avg_dead_ratio:.2f}",
                "최종 테스트 정확도 (%)": f"{final_test_acc:.2f}",
                "수렴 속도 (epochs)": convergence_epoch + 1,
                "평균 Gradient 크기": f"{avg_gradient:.6f}",
            }
        )

    df = pd.DataFrame(comparison_data)
    print("\n=== 정량적 비교 결과 ===")
    print(df.to_string(index=False))

    return df


def run_experiment_b():
    """실험 B 메인 실행 함수"""
    print("=" * 60)
    print("실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)")
    print("=" * 60)

    # GPU 설정
    device, use_gpu = setup_device()
    use_amp = use_gpu  # Mixed Precision은 GPU에서만 사용

    # 실험 설정
    epochs = 300
    learning_rate = 0.01 if use_gpu else 0.01

    # 2D 데이터셋 로드
    datasets = load_2d_datasets(use_gpu)

    # 활성화 함수들
    activation_functions = ["relu", "leaky_relu", "sigmoid"]
    activation_names = ["ReLU", "LeakyReLU", "Sigmoid"]

    for dataset_name, data in datasets.items():
        print(f"\n--- {dataset_name.upper()} 데이터셋으로 실험 ---")

        # GPU 메모리 정리
        clear_gpu_memory()

        results = {}
        gradients = {}

        for activation_fn, activation_name in zip(
            activation_functions, activation_names
        ):
            print(f"\n{activation_name} 활성화 함수로 학습 중...")

            # 모델 초기화
            model = MLPWithActivation(
                input_size=data["input_size"],
                num_classes=data["num_classes"],
                activation_fn=activation_fn,
                small_weights=True,  # Dead ReLU 유도를 위한 작은 가중치
            )

            # 모델 훈련 및 분석
            result = train_model_with_analysis(
                model, data, epochs, learning_rate, device, activation_name, use_amp
            )

            results[activation_name] = result

            # 그래디언트 분석
            gradient_analysis = analyze_gradients(model, data, device)
            gradients[activation_name] = gradient_analysis

            print(f"\n{activation_name} 그래디언트 분석:")
            for name, grad_val in gradient_analysis:
                print(f"  {name}: {grad_val:.6f}")

            # Dead ReLU 분석 (ReLU 계열에 대해서만)
            if activation_name in ["ReLU", "LeakyReLU"] and result["dead_relu_history"]:
                final_dead_ratios = result["dead_relu_history"][-1]
                print(f"\n{activation_name} 최종 Dead ReLU 비율:")
                for i, ratio in enumerate(final_dead_ratios):
                    print(f"  Layer {i+1}: {ratio:.2f}%")

            # 결정 경계 시각화
            plot_2d_decision_boundary(model, data, dataset_name, activation_name)

            # GPU 메모리 정리
            del model
            clear_gpu_memory()

        # 결과 시각화
        plot_learning_curves(results, dataset_name)
        plot_activation_distributions(results, dataset_name)
        plot_dead_relu_heatmap(results, dataset_name)

        # 정량적 비교
        comparison_df = create_comparison_table(results, gradients)

        # 결과를 CSV로 저장
        comparison_df.to_csv(
            f"experiment_b_comparison_{dataset_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    # 실험 B 실행
    run_experiment_b()

    print("\n" + "=" * 60)
    print("실험 B 완료!")
    print("결과 파일들:")
    print("- experiment_b_results_moons.png")
    print("- experiment_b_results_circles.png")
    print("- activation_distributions_moons.png")
    print("- activation_distributions_circles.png")
    print("- dead_relu_heatmap_moons.png")
    print("- dead_relu_heatmap_circles.png")
    print("- decision_boundary_*.png")
    print("- experiment_b_comparison_moons.csv")
    print("- experiment_b_comparison_circles.csv")
    print("=" * 60)
