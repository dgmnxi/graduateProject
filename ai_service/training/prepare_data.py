"""
데이터 전처리 및 준비 스크립트
이미지 → MediaPipe 관절값 → 학습 데이터셋 생성
"""

import sys
from pathlib import Path

# 부모 디렉토리를 python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.data_loader import DatasetBuilder, JointsBetasDataset


def prepare_datasets():
    """전체 데이터셋 준비"""
    
    # 경로 설정
    base_path = Path(__file__).parent.parent.parent / "data"
    project_root = Path(__file__).parent.parent.parent
    image_base = base_path / "raw" / "imageFiles"
    pkl_train = base_path / "raw" / "pkl" / "train"
    pkl_val = base_path / "raw" / "pkl" / "validation"
    pkl_test = base_path / "raw" / "pkl" / "test"
    
    processed_dir = base_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # MediaPipe 모델 경로
    model_path = str(project_root / "models" / "pose_landmarker.task")
    
    # 1. Training dataset
    print("\n" + "=" * 70)
    print("🔄 TRAINING Dataset Preparation")
    print("=" * 70)
    builder_train = DatasetBuilder(str(image_base), str(pkl_train), model_path)
    joints_train, betas_train = builder_train.build_dataset('train')
    builder_train.save_dataset(
        joints_train, betas_train, 
        str(processed_dir / "train_data.npz")
    )
    
    # 2. Validation dataset
    print("\n" + "=" * 70)
    print("🔄 VALIDATION Dataset Preparation")
    print("=" * 70)
    builder_val = DatasetBuilder(str(image_base), str(pkl_val), model_path)
    joints_val, betas_val = builder_val.build_dataset('validation')
    builder_val.save_dataset(
        joints_val, betas_val,
        str(processed_dir / "validation_data.npz")
    )
    
    # 3. Test dataset
    print("\n" + "=" * 70)
    print("🔄 TEST Dataset Preparation")
    print("=" * 70)
    builder_test = DatasetBuilder(str(image_base), str(pkl_test), model_path)
    joints_test, betas_test = builder_test.build_dataset('test')
    builder_test.save_dataset(
        joints_test, betas_test,
        str(processed_dir / "test_data.npz")
    )
    
    # 4. 데이터셋 통계 출력
    print("\n" + "=" * 70)
    print("📊 Dataset Statistics")
    print("=" * 70)
    
    joints_train, betas_train = JointsBetasDataset.load_dataset(
        str(processed_dir / "train_data.npz"))
    joints_val, betas_val = JointsBetasDataset.load_dataset(
        str(processed_dir / "validation_data.npz"))
    joints_test, betas_test = JointsBetasDataset.load_dataset(
        str(processed_dir / "test_data.npz"))
    
    print(f"\n📌 TRAIN:")
    print(f"   Samples: {len(joints_train):,}")
    print(f"   Joints: {joints_train.shape}")
    print(f"     Mean: {joints_train.mean():.4f}, Std: {joints_train.std():.4f}")
    print(f"     Min: {joints_train.min():.4f}, Max: {joints_train.max():.4f}")
    print(f"   Betas: {betas_train.shape}")
    print(f"     Mean: {betas_train.mean():.4f}, Std: {betas_train.std():.4f}")
    
    print(f"\n📌 VALIDATION:")
    print(f"   Samples: {len(joints_val):,}")
    print(f"   Joints: {joints_val.shape}")
    print(f"     Mean: {joints_val.mean():.4f}, Std: {joints_val.std():.4f}")
    print(f"   Betas: {betas_val.shape}")
    
    print(f"\n📌 TEST:")
    print(f"   Samples: {len(joints_test):,}")
    print(f"   Joints: {joints_test.shape}")
    print(f"     Mean: {joints_test.mean():.4f}, Std: {joints_test.std():.4f}")
    print(f"   Betas: {betas_test.shape}")
    
    print("\n" + "=" * 70)
    print("✅ All datasets prepared successfully!")
    print(f"📁 Location: {processed_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    prepare_datasets()
