"""
시각화 스크립트
train.py 실행 후 생성되는 결과 JSON을 시각화합니다.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def find_and_load_result(model_name: str):
    """
    모델명으로 가장 최신 결과를 자동으로 찾아 로드합니다.
    
    Args:
        model_name: 모델명 예시 "residualmlp"
        
    Returns:
        tuple: (dict: 결과 JSON 데이터, str: 사용된 run_name)
    """
    model_name = model_name.strip().lower()

    # visualize.py 위치: ai_service/outputs/visual/visualize.py
    # project_root는 graduateProject 루트가 되어야 함
    project_root = Path(__file__).resolve().parents[3]
    checkpoints_root = project_root / "ai_service" / "outputs" / "checkpoints"
    
    # 해당 모델의 모든 결과 폴더를 찾음
    model_folders = sorted(
        checkpoints_root.glob(f"*_{model_name}"),
        reverse=True  # 최신 순서
    )
    
    if not model_folders:
        print(f"❌ '{model_name}' 모델의 결과를 찾을 수 없습니다.")
        return None, None
    
    print(f"\n📂 '{model_name}' 모델의 사용 가능한 실행 결과:")
    print("="*60)
    for i, folder in enumerate(model_folders, 1):
        print(f"{i}. {folder.name}")
    print("="*60)
    
    # 가장 최신 결과 사용
    latest_folder = model_folders[0]
    result_path = latest_folder / f"{model_name}_results.json"
    
    if not result_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {result_path}")
        return None, None
    
    print(f"\n✓ 가장 최신 결과 사용: {latest_folder.name}")
    print(f"✓ 로드 중: {result_path}")
    
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    
    return result, latest_folder.name


def load_latest_results_for_models(model_names):
    """
    여러 모델의 최신 결과를 한 번에 로드합니다.

    Args:
        model_names: 예시 ["mlp", "transformer", "residualmlp", "gcn"]

    Returns:
        tuple: (results_by_model, run_name_by_model)
    """
    results_by_model = {}
    run_name_by_model = {}

    for model_name in model_names:
        result, run_name = find_and_load_result(model_name)
        if result is None:
            print(f"⚠️ '{model_name}' 최신 결과를 건너뜁니다.")
            continue
        results_by_model[model_name] = result
        run_name_by_model[model_name] = run_name

    return results_by_model, run_name_by_model


def plot_four_by_four_comparison(results_by_model, model_names):
    """
    4행 4열 비교 그래프를 그립니다.
    - 행: 기존 4개 그래프 타입 (Loss, MAE, Train Loss, Validation Loss)
    - 열: 모델 (mlp, transformer, residualmlp, gcn)
    """
    row_titles = ["Loss", "MAE", "Train Loss", "Validation Loss"]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex="row")
    fig.suptitle("Latest Training Comparison (4x4)", fontsize=16, fontweight="bold")

    for col_idx, model_name in enumerate(model_names):
        ax_col_title = model_name.upper()
        axes[0, col_idx].set_title(ax_col_title, fontsize=12, fontweight="bold")

        if model_name not in results_by_model:
            for row_idx in range(4):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=11, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            continue

        history = results_by_model[model_name]["history"]
        epochs = range(1, len(history["train_loss"]) + 1)

        # Row 1: Loss (train + val)
        ax = axes[0, col_idx]
        ax.plot(epochs, history["train_loss"], label="train_loss", marker="o", linewidth=1.5)
        ax.plot(epochs, history["val_loss"], label="val_loss", marker="s", linewidth=1.5)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(row_titles[0])
        if col_idx == len(model_names) - 1:
            ax.legend(fontsize=8, loc="best")

        # Row 2: MAE (train + val)
        ax = axes[1, col_idx]
        ax.plot(epochs, history["train_mae"], label="train_mae", marker="o", linewidth=1.5)
        ax.plot(epochs, history["val_mae"], label="val_mae", marker="s", linewidth=1.5)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(row_titles[1])
        if col_idx == len(model_names) - 1:
            ax.legend(fontsize=8, loc="best")

        # Row 3: Train Loss
        ax = axes[2, col_idx]
        ax.plot(epochs, history["train_loss"], color="tab:blue", marker="o", linewidth=1.8)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(row_titles[2])

        # Row 4: Validation Loss
        ax = axes[3, col_idx]
        ax.plot(epochs, history["val_loss"], color="tab:orange", marker="s", linewidth=1.8)
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel(row_titles[3])
        ax.set_xlabel("Epoch")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).parent / f"comparison_4x4_latest_{timestamp}.png"
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"✓ 4x4 비교 그래프 저장됨: {save_path}")
    plt.show()


def plot_training_curves(history, model_run_name):
    """
    학습 곡선 4종을 그립니다: Loss, MAE, Train Loss, Validation Loss
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Training Curves - {model_run_name}", fontsize=14, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="train_loss", marker='o')
    axes[0, 0].plot(epochs, history["val_loss"], label="val_loss", marker='s')
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(epochs, history["train_mae"], label="train_mae", marker='o')
    axes[0, 1].plot(epochs, history["val_mae"], label="val_mae", marker='s')
    axes[0, 1].set_title("MAE")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Train Loss
    axes[1, 0].plot(epochs, history["train_loss"], color="tab:blue", marker='o')
    axes[1, 0].set_title("Train Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Loss
    axes[1, 1].plot(epochs, history["val_loss"], color="tab:orange", marker='s')
    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 이미지 저장
    save_path = Path(__file__).parent / f"{model_run_name}_training_curves.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ 저장됨: {save_path}")
    plt.show()


def plot_per_beta_mae(metrics, model_run_name):
    """
    Beta별 MAE 막대 그래프를 그립니다.
    """
    per_beta_mae = metrics["per_beta_mae"]
    indices = np.arange(len(per_beta_mae))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(indices, per_beta_mae, color='steelblue', alpha=0.8)
    
    # 값을 막대 위에 표시
    for i, (idx, val) in enumerate(zip(indices, per_beta_mae)):
        ax.text(idx, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(indices)
    ax.set_xticklabels([f"beta_{i}" for i in indices], rotation=45)
    ax.set_xlabel("Beta Index")
    ax.set_ylabel("MAE")
    ax.set_title(f"Per Beta MAE - {model_run_name}")
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 이미지 저장
    save_path = Path(__file__).parent / f"{model_run_name}_per_beta_mae.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"✓ 저장됨: {save_path}")
    plt.show()


def print_metrics_summary(result):
    """
    결과 메트릭을 텍스트 형식으로 출력합니다.
    """
    metrics = result["metrics"]
    
    print("\n" + "="*60)
    print(f"📊 결과 요약: {result['run_name']}")
    print("="*60)
    print(f"모델: {result['model']}")
    print(f"상태: {result['status']}")
    print(f"\n지표 (Metrics):")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    
    print(f"\nBeta별 MAE:")
    for i, mae in enumerate(metrics['per_beta_mae']):
        print(f"  beta_{i:2d}: {mae:.6f}")
    
    mean_per_beta = np.mean(metrics['per_beta_mae'])
    print(f"  평균:   {mean_per_beta:.6f}")
    print("="*60 + "\n")


def save_metrics_to_text(result, model_run_name):
    """
    메트릭을 텍스트 파일로 저장합니다.
    """
    metrics = result["metrics"]
    
    output_file = Path(__file__).parent / f"{model_run_name}_metrics.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"결과 요약\n")
        f.write(f"{'='*60}\n")
        f.write(f"모델: {result['model']}\n")
        f.write(f"Run Name: {result['run_name']}\n")
        f.write(f"상태: {result['status']}\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n지표 (Metrics)\n")
        f.write(f"{'-'*60}\n")
        f.write(f"MSE:  {metrics['mse']:.10f}\n")
        f.write(f"MAE:  {metrics['mae']:.10f}\n")
        f.write(f"RMSE: {metrics['rmse']:.10f}\n")
        f.write(f"R²:   {metrics['r2']:.2f}\n")
        
        f.write(f"\nBeta별 MAE\n")
        f.write(f"{'-'*60}\n")
        for i, mae in enumerate(metrics['per_beta_mae']):
            f.write(f"beta_{i:2d}: {mae:.10f}\n")
        
        mean_per_beta = np.mean(metrics['per_beta_mae'])
        f.write(f"{'평균':>8}: {mean_per_beta:.10f}\n")
    
    print(f"✓ 저장됨: {output_file}")


def main():
    """
    메인 함수
    """
    print("\n" + "="*60)
    print("📈 시각화 스크립트 (4모델 최신 비교 모드)")
    print("="*60)

    model_names = ["mlp", "transformer", "residualmlp", "gcn"]
    print("\n대상 모델:", ", ".join(model_names))
    print("각 모델별 최신 결과를 자동으로 로드합니다.")

    results_by_model, run_name_by_model = load_latest_results_for_models(model_names)
    if not results_by_model:
        print("❌ 로드 가능한 결과가 없습니다.")
        return

    print("\n📌 로드된 최신 run_name")
    print("="*60)
    for model_name in model_names:
        run_name = run_name_by_model.get(model_name)
        if run_name is None:
            print(f"- {model_name}: 없음")
        else:
            print(f"- {model_name}: {run_name}")
    print("="*60)

    print("\n📊 모델별 메트릭 요약")
    for model_name in model_names:
        if model_name in results_by_model:
            print_metrics_summary(results_by_model[model_name])

    print("📉 4x4 비교 그래프 생성 중...")
    plot_four_by_four_comparison(results_by_model, model_names)

    for model_name in model_names:
        if model_name in results_by_model:
            save_metrics_to_text(results_by_model[model_name], run_name_by_model[model_name])

    print("\n✓ 완료!")


if __name__ == "__main__":
    main()
