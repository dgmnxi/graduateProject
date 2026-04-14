"""
MediaPipe 기반 데이터 로더
이미지 → MediaPipe 33개 관절값 추출 → SMPL beta 값과 매칭
"""

import os
import cv2
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

# MediaPipe 0.10+ 새로운 API 사용
try:
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False


class MediaPipeProcessor:
    """MediaPipe를 사용한 이미지 처리 (MediaPipe 0.10+ API)"""
    
    def __init__(self, model_path: str = "models/pose_landmarker.task"):
        if not MEDIAPIPE_AVAILABLE:
            print("Warning: MediaPipe not available!")
            self.pose_landmarker = None
            return
        
        try:
            # Resolve model path
            if not Path(model_path).is_absolute():
                # Make relative path absolute from project root
                model_path = str(Path(__file__).parent.parent.parent / model_path)
            
            # MediaPipe Task API - create_from_model_path 사용
            self.pose_landmarker = vision.PoseLandmarker.create_from_model_path(model_path)
        except Exception as e:
            print(f"Failed to initialize PoseLandmarker: {e}")
            self.pose_landmarker = None
    
    def extract_landmarks(self, image_path: str) -> np.ndarray:
        """
        이미지에서 MediaPipe 33개 관절값 추출
        
        Returns:
            np.ndarray: shape (99,) = 33개 관절 × 3좌표 (x, y, z)
        """
        if self.pose_landmarker is None:
            return np.zeros(99)
        
        try:
            # 이미지 읽기
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros(99)
            
            # MediaPipe 입력 형식으로 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Pose 감지
            detection_result = self.pose_landmarker.detect(mp_image)
            
            landmarks = np.zeros(99)
            
            # 첫 번째 person의 landmarks 추출
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks_list = detection_result.pose_landmarks[0]
                for idx, landmark in enumerate(landmarks_list):
                    if idx < 33:  # MediaPipe는 33개 landmarks
                        landmarks[idx * 3] = landmark.x
                        landmarks[idx * 3 + 1] = landmark.y
                        landmarks[idx * 3 + 2] = landmark.z
            
            return landmarks
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(99)
    
    def __del__(self):
        if hasattr(self, 'pose_landmarker') and self.pose_landmarker is not None:
            try:
                # 리소스 정리
                pass
            except:
                pass


class DatasetBuilder:
    """학습 데이터셋 구축"""
    
    def __init__(self, image_dir: str, pkl_dir: str, model_path: str = "models/pose_landmarker.task"):
        self.image_dir = Path(image_dir)
        self.pkl_dir = Path(pkl_dir)
        self.processor = MediaPipeProcessor(model_path)
        self.data = []
    
    def build_dataset(self, split_name: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터셋 구축
        
        Args:
            split_name: 'train', 'validation', 'test' 중 하나
        
        Returns:
            joints: (N, 99) - MediaPipe 관절값
            betas: (N, 10) - SMPL beta 값
        """
        joints_list = []
        betas_list = []
        
        # 해당 split의 pkl 파일 목록 가져오기
        pkl_files = sorted(self.pkl_dir.glob('*.pkl'))
        
        # pkl 파일들에 대한 진행 바
        for pkl_file in tqdm(pkl_files, desc=f'Loading {split_name} sequences', unit='seq'):
            sequence_name = pkl_file.stem  # 파일명 (확장자 제외)
            image_folder = self.image_dir / sequence_name
            
            if not image_folder.exists():
                tqdm.write(f"Warning: Image folder not found for {sequence_name}")
                continue
            
            # pkl 파일 로드
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
            except Exception as e:
                tqdm.write(f"Error loading {pkl_file}: {e}")
                continue
            
            # 2명의 인물에 대해 처리
            num_people = len(data['betas'])
            img_frame_ids = data['img_frame_ids']
            
            for person_idx in range(num_people):
                beta = data['betas'][person_idx]  # numpy array shape (10,)
                
                # 각 이미지 프레임 처리 (진행 바)
                for img_frame_id in tqdm(img_frame_ids, desc=f'  {sequence_name} (person {person_idx+1}/{num_people})', 
                                        leave=False, unit='frame'):
                    image_path = image_folder / f"image_{img_frame_id:05d}.jpg"
                    
                    if not image_path.exists():
                        continue
                    
                    # MediaPipe로 관절값 추출
                    joints = self.processor.extract_landmarks(str(image_path))
                    
                    # 데이터 저장 (float32로 정규화)
                    joints_list.append(np.asarray(joints, dtype=np.float32))
                    betas_list.append(np.asarray(beta, dtype=np.float32))
        
        # 리스트가 비어있나 확인
        if len(joints_list) == 0 or len(betas_list) == 0:
            print(f"⚠ Warning: No data found for {split_name}")
            print(f"  joints_list: {len(joints_list)}, betas_list: {len(betas_list)}")
            return np.zeros((0, 99)), np.zeros((0, 10))
        
        # 길이 확인
        if len(joints_list) != len(betas_list):
            print(f"⚠ Warning: Length mismatch! joints: {len(joints_list)}, betas: {len(betas_list)}")
            # 짧은 쪽에 맞추기
            min_len = min(len(joints_list), len(betas_list))
            joints_list = joints_list[:min_len]
            betas_list = betas_list[:min_len]
        
        # 각 배열이 정확한 shape인지 확인하며 변환
        joints_array = []
        betas_array = []
        
        for j, b in zip(joints_list, betas_list):
            j_arr = np.asarray(j, dtype=np.float32).reshape(-1)
            b_arr = np.asarray(b, dtype=np.float32).reshape(-1)
            
            # shape 확인
            if j_arr.shape[0] == 99 and b_arr.shape[0] == 10:
                joints_array.append(j_arr)
                betas_array.append(b_arr)
        
        # 최종 배열로 변환
        joints = np.array(joints_array, dtype=np.float32)  # (N, 99)
        betas = np.array(betas_array, dtype=np.float32)    # (N, 10)
        
        print(f"\n✓ Dataset {split_name}: {len(joints)} samples")
        print(f"  Joints shape: {joints.shape}")
        print(f"  Betas shape: {betas.shape}")
        
        return joints, betas
    
    def save_dataset(self, joints: np.ndarray, betas: np.ndarray, 
                    output_path: str):
        """
        데이터셋 저장
        """
        np.savez(output_path, joints=joints, betas=betas)
        print(f"Dataset saved to {output_path}")
    
    @staticmethod
    def load_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        저장된 데이터셋 로드
        """
        data = np.load(file_path, allow_pickle=True)
        return data['joints'], data['betas']


class JointsBetasDataset:
    """PyTorch Dataset (선택사항)"""
    
    def __init__(self, joints: np.ndarray, betas: np.ndarray):
        self.joints = torch.FloatTensor(joints)
        self.betas = torch.FloatTensor(betas)
    
    def __len__(self):
        return len(self.joints)
    
    def __getitem__(self, idx):
        return self.joints[idx], self.betas[idx]
    
    @staticmethod
    def load_dataset(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        저장된 데이터셋 로드
        """
        data = np.load(file_path, allow_pickle=True)
        return data['joints'], data['betas']
