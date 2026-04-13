"""
Stage 1 preprocessing utilities.

Scope:
- Missing value handling
- Coordinate normalization
- Outlier clipping
- Lightweight feature engineering
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

NUM_JOINTS = 33
COORD_DIM = 3
EXPECTED_INPUT_SIZE = NUM_JOINTS * COORD_DIM
EPS = 1e-6

# MediaPipe Pose indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass
class PreprocessConfig:
    normalization: str = "zscore"  # zscore | minmax
    clip_zscore: float = 3.0
    center_on_pelvis: bool = True
    include_engineered_features: bool = True


def _reshape_joints(joints: np.ndarray) -> np.ndarray:
    """Validate and reshape incoming joints into (33, 3)."""
    joints = np.asarray(joints, dtype=np.float32)

    if joints.ndim == 1:
        if joints.shape[0] != EXPECTED_INPUT_SIZE:
            raise ValueError(
                f"Expected flat joints with {EXPECTED_INPUT_SIZE} values, got {joints.shape[0]}"
            )
        return joints.reshape(NUM_JOINTS, COORD_DIM)

    if joints.ndim == 2 and joints.shape == (NUM_JOINTS, COORD_DIM):
        return joints

    raise ValueError(
        "Joints must be shape (99,) or (33, 3). "
        f"Received shape {tuple(joints.shape)}"
    )


def _fill_missing(joints: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fill NaN values with per-axis mean, then 0 if entire axis is missing."""
    out = joints.copy()
    missing_mask = np.isnan(out)
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return out, 0

    for axis in range(COORD_DIM):
        axis_values = out[:, axis]
        axis_mean = np.nanmean(axis_values)
        if np.isnan(axis_mean):
            axis_mean = 0.0
        axis_values[np.isnan(axis_values)] = axis_mean
        out[:, axis] = axis_values

    return out, missing_count


def _center_on_pelvis(joints: np.ndarray) -> np.ndarray:
    pelvis = (joints[LEFT_HIP] + joints[RIGHT_HIP]) / 2.0
    return joints - pelvis


def _normalize(joints: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zscore":
        mean = joints.mean(axis=0, keepdims=True)
        std = joints.std(axis=0, keepdims=True)
        std = np.where(std < EPS, 1.0, std)
        return (joints - mean) / std

    if mode == "minmax":
        min_v = joints.min(axis=0, keepdims=True)
        max_v = joints.max(axis=0, keepdims=True)
        denom = np.where((max_v - min_v) < EPS, 1.0, max_v - min_v)
        return (joints - min_v) / denom

    raise ValueError(f"Unsupported normalization mode: {mode}")


def _clip_outliers_zscore(joints: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    mean = joints.mean(axis=0, keepdims=True)
    std = joints.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)

    z = (joints - mean) / std
    clipped_z = np.clip(z, -threshold, threshold)
    clipped = mean + clipped_z * std
    clipped_count = int(np.sum(np.abs(z) > threshold))

    return clipped, clipped_count


def _distance(joints: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(joints[a] - joints[b]))


def _angle(joints: np.ndarray, a: int, b: int, c: int) -> float:
    """Angle ABC in radians."""
    v1 = joints[a] - joints[b]
    v2 = joints[c] - joints[b]

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return 0.0

    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def extract_features(joints: np.ndarray) -> np.ndarray:
    """Extract compact geometric features from normalized joints."""
    features = []

    # Segment distances
    distance_pairs: Iterable[Tuple[int, int]] = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    for a, b in distance_pairs:
        features.append(_distance(joints, a, b))

    # Joint angles
    angle_triplets: Iterable[Tuple[int, int, int]] = [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ]
    for a, b, c in angle_triplets:
        features.append(_angle(joints, a, b, c))

    # Symmetry and ratio features
    left_arm = _distance(joints, LEFT_SHOULDER, LEFT_ELBOW) + _distance(joints, LEFT_ELBOW, LEFT_WRIST)
    right_arm = _distance(joints, RIGHT_SHOULDER, RIGHT_ELBOW) + _distance(joints, RIGHT_ELBOW, RIGHT_WRIST)
    left_leg = _distance(joints, LEFT_HIP, LEFT_KNEE) + _distance(joints, LEFT_KNEE, LEFT_ANKLE)
    right_leg = _distance(joints, RIGHT_HIP, RIGHT_KNEE) + _distance(joints, RIGHT_KNEE, RIGHT_ANKLE)

    shoulder_width = _distance(joints, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_width = _distance(joints, LEFT_HIP, RIGHT_HIP)
    torso_height = _distance(joints, LEFT_SHOULDER, LEFT_HIP)

    features.extend(
        [
            abs(left_arm - right_arm),
            abs(left_leg - right_leg),
            shoulder_width / max(hip_width, EPS),
            (left_arm + right_arm) / max(left_leg + right_leg, EPS),
            torso_height / max(hip_width, EPS),
        ]
    )

    return np.asarray(features, dtype=np.float32)


def preprocess_pose(joints: np.ndarray, config: PreprocessConfig) -> Dict[str, object]:
    """
    End-to-end preprocessing for Stage 1 baseline.

    Returns:
        normalized_joints: (33, 3)
        feature_vector: engineered features only
        model_input: flattened joints + engineered features
        metadata: processing statistics
    """
    joint_matrix = _reshape_joints(joints)
    joint_matrix, missing_count = _fill_missing(joint_matrix)

    if config.center_on_pelvis:
        joint_matrix = _center_on_pelvis(joint_matrix)

    normalized = _normalize(joint_matrix, config.normalization)
    normalized, clipped_count = _clip_outliers_zscore(normalized, config.clip_zscore)

    feature_vector = extract_features(normalized) if config.include_engineered_features else np.empty((0,), dtype=np.float32)
    model_input = np.concatenate([normalized.reshape(-1), feature_vector], axis=0)

    return {
        "normalized_joints": normalized.astype(np.float32),
        "feature_vector": feature_vector.astype(np.float32),
        "model_input": model_input.astype(np.float32),
        "metadata": {
            "input_size": int(EXPECTED_INPUT_SIZE),
            "normalized_size": int(normalized.size),
            "feature_size": int(feature_vector.size),
            "missing_filled": missing_count,
            "outliers_clipped": clipped_count,
            "normalization": config.normalization,
            "center_on_pelvis": config.center_on_pelvis,
        },
    }
