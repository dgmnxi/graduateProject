"""
모델 학습 스크립트
"""

import sys
import logging
from datetime import datetime
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

from training.models import create_model


MODEL_ALIASES = {
    'resmlp': 'residualmlp'
}

ALL_MODELS = ['mlp', 'transformer', 'residualmlp', 'gcn']

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / 'outputs'


class TrainingOOMError(RuntimeError):
    """CUDA OOM 발생 시 학습 루프를 안전하게 중단하기 위한 예외"""


def normalize_model_name(model_name: str) -> str:
    """사용자 입력 모델명을 내부 표준명으로 정규화"""
    model_name = model_name.lower().strip()
    return MODEL_ALIASES.get(model_name, model_name)


def is_cuda_oom_error(error: RuntimeError) -> bool:
    """PyTorch CUDA OOM 오류 여부 확인"""
    message = str(error).lower()
    return 'cuda out of memory' in message or 'cublas_status_alloc_failed' in message


def cleanup_cuda_memory():
    """CUDA 캐시 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_timestamp(include_time: bool = True) -> str:
    """실행 시각 문자열 생성"""
    if include_time:
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    return datetime.now().strftime('%Y%m%d')


def build_run_name(model_name: str, run_suffix_mode: str = 'auto', custom_suffix: str = '', timestamp: str | None = None) -> str:
    """run_name 생성: YYYYMMDD[_HHMMSS]_MODEL"""
    normalized_model = normalize_model_name(model_name)
    base_timestamp = timestamp or get_timestamp(include_time=(run_suffix_mode != 'none'))

    if run_suffix_mode == 'none':
        return f"{base_timestamp}_{normalized_model}"
    if run_suffix_mode == 'custom' and custom_suffix:
        return f"{base_timestamp}_{custom_suffix}_{normalized_model}"
    return f"{base_timestamp}_{normalized_model}"


def setup_logger(run_name: str, logs_dir: Path) -> logging.Logger:
    """콘솔 + 파일 로거 초기화"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler(logs_dir / f'{run_name}.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    return logger


def create_output_dirs(outputs_root: Path):
    """outputs 하위 기본 디렉토리 생성"""
    checkpoints_root = outputs_root / 'checkpoints'
    logs_root = outputs_root / 'logs'
    visual_root = outputs_root / 'visual'
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)
    return checkpoints_root, logs_root, visual_root


def resolve_run_paths(outputs_root: Path, model_name: str, run_suffix_mode: str, custom_suffix: str = '', timestamp: str | None = None):
    """모델별 run_name/경로 계산"""
    checkpoints_root, logs_root, visual_root = create_output_dirs(outputs_root)
    run_name = build_run_name(model_name, run_suffix_mode, custom_suffix, timestamp=timestamp)
    model_run_dir = checkpoints_root / run_name
    model_run_dir.mkdir(parents=True, exist_ok=True)
    return {
        'run_name': run_name,
        'checkpoints_root': checkpoints_root,
        'logs_root': logs_root,
        'visual_root': visual_root,
        'model_run_dir': model_run_dir,
    }


class Trainer:
    """모델 학습기"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', logger=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.best_val_loss = float('inf')
        self.logger = logger or logging.getLogger(__name__)
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

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    mae = torch.mean(torch.abs(outputs - targets))

                total_loss += loss.item()
                total_mae += mae.item()

                loss_val = loss.item()
                mae_val = mae.item()
                pbar.set_postfix({'loss': f'{loss_val:.4f}', 'mae': f'{mae_val:.4f}'})

                if batch_idx % 100 == 0:
                    self.logger.info(f'Training step {batch_idx}: loss={loss_val:.4f}, mae={mae_val:.4f}')
            except RuntimeError as error:
                if is_cuda_oom_error(error):
                    optimizer.zero_grad(set_to_none=True)
                    cleanup_cuda_memory()
                    raise TrainingOOMError(
                        f'CUDA OOM in train_epoch at batch {batch_idx}. Try smaller --batch_size or lighter model settings.'
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

                    if batch_idx % 100 == 0:
                        self.logger.info(f'Validation step {batch_idx}: loss={loss_val:.4f}, mae={mae_val:.4f}')
                except RuntimeError as error:
                    if is_cuda_oom_error(error):
                        cleanup_cuda_memory()
                        raise TrainingOOMError(
                            f'CUDA OOM in validate at batch {batch_idx}. Try smaller --batch_size.'
                        ) from error
                    raise

        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / len(val_loader)

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        return avg_loss, avg_mae, predictions, targets

    def train(self, train_loader, val_loader, epochs=100, lr=0.001, weight_decay=0.0001, save_dir='./checkpoints'):
        """전체 학습 루프"""

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            try:
                self.logger.info(f'Epoch {epoch}/{epochs}')

                train_loss, train_mae = self.train_epoch(train_loader, optimizer, scheduler)
                self.history['train_loss'].append(train_loss)
                self.history['train_mae'].append(train_mae)

                val_loss, val_mae, predictions, targets = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)

                self.logger.info(f'Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}')
                self.logger.info(f'Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}')

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    checkpoint_path = Path(save_dir) / f'{self.model.__class__.__name__}_best.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    self.logger.info(f'Best model saved to {checkpoint_path}')
            except TrainingOOMError as oom_error:
                checkpoint_path = Path(save_dir) / f'{self.model.__class__.__name__}_oom_epoch{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': self.history,
                    'error': str(oom_error),
                }, checkpoint_path)
                self.logger.info(f'OOM checkpoint saved to {checkpoint_path}')
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
                            f'CUDA OOM in evaluate at batch {batch_idx}. Try smaller --batch_size.'
                        ) from error
                    raise

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'per_beta_mae': [float(np.mean(np.abs(predictions[:, i] - targets[:, i]))) for i in range(targets.shape[1])]
        }

        return metrics, predictions


def load_datasets(data_dir='./data/processed', batch_size=32):
    """데이터셋 로드"""

    train_data = np.load(Path(data_dir) / 'train_data.npz')
    joints_train, betas_train = train_data['joints'], train_data['betas']

    val_data = np.load(Path(data_dir) / 'validation_data.npz')
    joints_val, betas_val = val_data['joints'], val_data['betas']

    test_data = np.load(Path(data_dir) / 'test_data.npz')
    joints_test, betas_test = test_data['joints'], test_data['betas']

    mean = joints_train.mean(axis=0)
    std = joints_train.std(axis=0)
    std[std == 0] = 1

    joints_train = (joints_train - mean) / std
    joints_val = (joints_val - mean) / std
    joints_test = (joints_test - mean) / std

    train_dataset = TensorDataset(torch.FloatTensor(joints_train), torch.FloatTensor(betas_train))
    val_dataset = TensorDataset(torch.FloatTensor(joints_val), torch.FloatTensor(betas_val))
    test_dataset = TensorDataset(torch.FloatTensor(joints_test), torch.FloatTensor(betas_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Train samples: {len(joints_train)}')
    print(f'Val samples: {len(joints_val)}')
    print(f'Test samples: {len(joints_test)}')

    return train_loader, val_loader, test_loader


def train_and_evaluate_single_model(model_name: str, args, train_loader, val_loader, test_loader, timestamp: str):
    """단일 모델 생성/학습/평가/결과 저장"""
    normalized_model = normalize_model_name(model_name)

    paths = resolve_run_paths(
        outputs_root=Path(args.outputs_root),
        model_name=normalized_model,
        run_suffix_mode=args.run_suffix,
        custom_suffix=args.run_suffix_value,
        timestamp=timestamp,
    )
    run_name = paths['run_name']
    model_save_dir = paths['model_run_dir']
    logger = setup_logger(run_name, paths['logs_root'])

    logger.info(f'Creating {normalized_model.upper()} model...')
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

    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    trainer = Trainer(model, logger=logger)
    status = 'success'
    metrics = None
    oom_error = None
    history = trainer.history

    try:
        logger.info(f'Training {normalized_model.upper()} model...')
        history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=str(model_save_dir)
        )

        logger.info(f'Evaluating {normalized_model.upper()} model...')
        metrics, predictions = trainer.evaluate(test_loader)

        logger.info('=== Test Results ===')
        logger.info(f"MSE: {metrics['mse']:.6f}")
        logger.info(f"MAE: {metrics['mae']:.6f}")
        logger.info(f"RMSE: {metrics['rmse']:.6f}")
        logger.info(f"R²: {metrics['r2']:.6f}")
        logger.info('Per-Beta MAE:')
        for i, mae_val in enumerate(metrics['per_beta_mae']):
            logger.info(f'  Beta[{i}]: {mae_val:.6f}')
    except TrainingOOMError as error:
        status = 'failed_oom'
        oom_error = str(error)
        history = trainer.history
        logger.info(f'OOM detected: {oom_error}')
        logger.info('Training stopped safely after CUDA memory cleanup.')
    finally:
        cleanup_cuda_memory()

    result_path = model_save_dir / f'{normalized_model}_results.json'
    try:
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': normalized_model,
                'run_name': run_name,
                'status': status,
                'oom_error': oom_error,
                'metrics': metrics,
                'history': history
            }, f, indent=2)

        logger.info(f'Results saved to {result_path}')
    finally:
        for handler in list(logger.handlers):
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

    return {
        'model': normalized_model,
        'run_name': run_name,
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
    parser.add_argument('--outputs_root', type=str, default=str(DEFAULT_OUTPUTS_ROOT))
    parser.add_argument('--run_suffix', type=str, default='auto', choices=['auto', 'none', 'custom'])
    parser.add_argument('--run_suffix_value', type=str, default='')
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    create_output_dirs(outputs_root)

    try:
        print('Loading datasets...')
        train_loader, val_loader, test_loader = load_datasets(args.data_dir, args.batch_size)

        normalized_target = normalize_model_name(args.model)
        main_timestamp = get_timestamp(include_time=True)

        if normalized_target == 'all':
            print('\nRunning all models sequentially...')
            summaries = []
            for model_name in ALL_MODELS:
                print(f"\n{'=' * 60}")
                print(f'Starting model: {model_name}')
                print(f"{'=' * 60}")
                summary = train_and_evaluate_single_model(
                    model_name, args, train_loader, val_loader, test_loader, timestamp=main_timestamp
                )
                summaries.append(summary)

            all_run_name = build_run_name('all', args.run_suffix, args.run_suffix_value, timestamp=main_timestamp)
            summary_dir = outputs_root / 'checkpoints' / all_run_name
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_path = summary_dir / 'all_models_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({'models': summaries}, f, indent=2)

            print(f'\nAll model summary saved to {summary_path}')
        else:
            train_and_evaluate_single_model(
                normalized_target, args, train_loader, val_loader, test_loader, timestamp=main_timestamp
            )
    finally:
        cleanup_cuda_memory()


if __name__ == '__main__':
    main()
