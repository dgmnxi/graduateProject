"""
모델 학습 스크립트
"""

import sys
from pathlib import Path

# 부모 디렉토리를 python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
import argparse
from typing import Tuple, Dict
import time

from training.models import create_model


MODEL_ALIASES = {
    'resmlp': 'residualmlp'
}

ALL_MODELS = ['mlp', 'transformer', 'residualmlp', 'gcn']


class TrainingOOMError(RuntimeError):
    """CUDA OOM 발생 시 학습 루프를 안전하게 중단하기 위한 예외"""


def is_cuda_oom_error(error: RuntimeError) -> bool:
    """PyTorch CUDA OOM 오류 여부 확인"""
    message = str(error).lower()
    return 'cuda out of memory' in message or 'cublas_status_alloc_failed' in message


def cleanup_cuda_memory():
    """CUDA 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Trainer:
    """모델 학습기"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                # 메트릭 계산
                with torch.no_grad():
                    mae = torch.mean(torch.abs(outputs - targets))

                total_loss += loss.item()
                total_mae += mae.item()

                loss_val = loss.item()
                mae_val = mae.item()
                pbar.set_postfix({'loss': f'{loss_val:.4f}', 'mae': f'{mae_val:.4f}'})
            except RuntimeError as error:
                if is_cuda_oom_error(error):
                    optimizer.zero_grad(set_to_none=True)
                    cleanup_cuda_memory()
                    raise TrainingOOMError(
                        f"CUDA OOM in train_epoch at batch {batch_idx}. "
                        f"Try smaller --batch_size or lighter model settings."
                    ) from error
                raise
        
        avg_loss = total_loss / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        
        if scheduler is not None:
            scheduler.step()
        
        return avg_loss, avg_mae
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                try:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    mae = torch.mean(torch.abs(outputs - targets))

                    total_loss += loss.item()
                    total_mae += mae.item()

                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())

                    loss_val = loss.item()
                    mae_val = mae.item()
                    pbar.set_postfix({'loss': f'{loss_val:.4f}', 'mae': f'{mae_val:.4f}'})
                except RuntimeError as error:
                    if is_cuda_oom_error(error):
                        cleanup_cuda_memory()
                        raise TrainingOOMError(
                            f"CUDA OOM in validate at batch {batch_idx}. "
                            f"Try smaller --batch_size."
                        ) from error
                    raise
        
        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / len(val_loader)
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return avg_loss, avg_mae, predictions, targets
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, 
              weight_decay=0.0001, save_dir='./checkpoints'):
        """전체 학습 루프"""
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(1, epochs + 1):
            try:
                print(f"\nEpoch {epoch}/{epochs}")

                # 학습
                train_loss, train_mae = self.train_epoch(train_loader, optimizer, scheduler)
                self.history['train_loss'].append(train_loss)
                self.history['train_mae'].append(train_mae)

                # 검증
                val_loss, val_mae, predictions, targets = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)

                print(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
                print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

                # Best model 저장
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = Path(save_dir) / f"{self.model.__class__.__name__}_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"Best model saved to {checkpoint_path}")

                # 주기적 저장
                if epoch % 10 == 0:
                    checkpoint_path = Path(save_dir) / f"{self.model.__class__.__name__}_epoch{epoch}.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
            except TrainingOOMError as oom_error:
                checkpoint_path = Path(save_dir) / f"{self.model.__class__.__name__}_oom_epoch{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': self.history,
                    'error': str(oom_error),
                }, checkpoint_path)
                print(f"OOM checkpoint saved to {checkpoint_path}")
                cleanup_cuda_memory()
                raise
        
        return self.history
    
    def evaluate(self, test_loader):
        """테스트셋 평가"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc='Testing')):
                try:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(targets.cpu().numpy())
                except RuntimeError as error:
                    if is_cuda_oom_error(error):
                        cleanup_cuda_memory()
                        raise TrainingOOMError(
                            f"CUDA OOM in evaluate at batch {batch_idx}. "
                            f"Try smaller --batch_size."
                        ) from error
                    raise
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # 평가 지표 계산
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # R² 계산
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'per_beta_mae': [float(np.mean(np.abs(predictions[:, i] - targets[:, i]))) 
                            for i in range(targets.shape[1])]
        }
        
        return metrics, predictions


def load_datasets(data_dir='./data/processed', batch_size=32):
    """데이터셋 로드"""
    
    # 데이터 로드
    train_data = np.load(Path(data_dir) / 'train_data.npz')
    joints_train, betas_train = train_data['joints'], train_data['betas']
    
    val_data = np.load(Path(data_dir) / 'validation_data.npz')
    joints_val, betas_val = val_data['joints'], val_data['betas']
    
    test_data = np.load(Path(data_dir) / 'test_data.npz')
    joints_test, betas_test = test_data['joints'], test_data['betas']
    
    # 정규화
    mean = joints_train.mean(axis=0)
    std = joints_train.std(axis=0)
    std[std == 0] = 1  # 0으로 나누기 방지
    
    joints_train = (joints_train - mean) / std
    joints_val = (joints_val - mean) / std
    joints_test = (joints_test - mean) / std
    
    # DataLoader 생성
    train_dataset = TensorDataset(
        torch.FloatTensor(joints_train),
        torch.FloatTensor(betas_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(joints_val),
        torch.FloatTensor(betas_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(joints_test),
        torch.FloatTensor(betas_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(joints_train)}")
    print(f"Val samples: {len(joints_val)}")
    print(f"Test samples: {len(joints_test)}")
    
    return train_loader, val_loader, test_loader


def normalize_model_name(model_name: str) -> str:
    """사용자 입력 모델명을 내부 표준명으로 정규화"""
    model_name = model_name.lower().strip()
    return MODEL_ALIASES.get(model_name, model_name)


def train_and_evaluate_single_model(model_name: str, args, train_loader, val_loader, test_loader):
    """단일 모델 생성/학습/평가/결과 저장"""
    normalized_model = normalize_model_name(model_name)

    # all 모드에서는 모델별 하위 폴더에 저장
    if normalize_model_name(args.model) == 'all':
        model_save_dir = str(Path(args.save_dir) / normalized_model)
    else:
        model_save_dir = args.save_dir

    print(f"\nCreating {normalized_model.upper()} model...")
    model = create_model(
        normalized_model,
        input_size=99,
        output_size=10,
        hidden_sizes=[256, 128, 64],
        dropout_rates=[0.2, 0.2, 0.1],
        feat_dim=64,
        hidden_dim=128,
        num_layers=3
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model)
    status = 'success'
    metrics = None
    oom_error = None

    try:
        print(f"\nTraining {normalized_model.upper()} model...")
        history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=model_save_dir
        )

        print(f"\nEvaluating {normalized_model.upper()} model...")
        metrics, predictions = trainer.evaluate(test_loader)

        print("\n=== Test Results ===")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        print(f"\nPer-Beta MAE:")
        for i, mae_val in enumerate(metrics['per_beta_mae']):
            print(f"  Beta[{i}]: {mae_val:.6f}")
    except TrainingOOMError as error:
        status = 'failed_oom'
        oom_error = str(error)
        history = trainer.history
        print(f"\nOOM detected: {oom_error}")
        print("Training stopped safely after CUDA memory cleanup.")
    finally:
        cleanup_cuda_memory()

    result_path = Path(model_save_dir) / f"{normalized_model}_results.json"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump({
            'model': normalized_model,
            'status': status,
            'oom_error': oom_error,
            'metrics': metrics,
            'history': history
        }, f, indent=2)

    print(f"\nResults saved to {result_path}")

    return {
        'model': normalized_model,
        'status': status,
        'oom_error': oom_error,
        'metrics': metrics,
        'result_path': str(result_path)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='residualmlp',
        choices=['mlp', 'transformer', 'residualmlp', 'resmlp', 'gcn', 'all']
    )
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    try:
        # 데이터셋 로드
        print("Loading datasets...")
        train_loader, val_loader, test_loader = load_datasets(args.data_dir, args.batch_size)

        normalized_target = normalize_model_name(args.model)

        if normalized_target == 'all':
            print("\nRunning all models sequentially...")
            summaries = []
            for model_name in ALL_MODELS:
                print(f"\n{'=' * 60}")
                print(f"Starting model: {model_name}")
                print(f"{'=' * 60}")
                summary = train_and_evaluate_single_model(
                    model_name, args, train_loader, val_loader, test_loader
                )
                summaries.append(summary)

            summary_path = Path(args.save_dir) / "all_models_summary.json"
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump({'models': summaries}, f, indent=2)

            print(f"\nAll model summary saved to {summary_path}")
        else:
            train_and_evaluate_single_model(
                normalized_target, args, train_loader, val_loader, test_loader
            )
    finally:
        cleanup_cuda_memory()


if __name__ == "__main__":
    main()
