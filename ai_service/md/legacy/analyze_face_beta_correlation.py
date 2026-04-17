"""
얼굴 면적 기반 feature와 SMPL beta 간 상관 분석 스크립트.

사용 예시:
python training/analyze_face_beta_correlation.py \
  --data data/processed/train_data.npz data/processed/validation_data.npz data/processed/test_data.npz \
  --output ai_service/outputs/visual/face_beta_correlation_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# MediaPipe Pose landmark index (33개)
FACE_LANDMARKS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

EPS = 1e-8


def resolve_existing_input_path(raw_path: Path, project_root: Path, ai_service_root: Path) -> Path | None:
    """입력 파일 경로를 실행 위치와 무관하게 탐색해서 반환."""
    candidates = [
        raw_path,
        Path.cwd() / raw_path,
        project_root / raw_path,
        ai_service_root / raw_path,
    ]

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def resolve_output_path(raw_path: Path, project_root: Path) -> Path:
    """출력 경로는 프로젝트 루트 기준으로 고정 해석."""
    if raw_path.is_absolute():
        return raw_path
    return (project_root / raw_path).resolve()


def ensure_joints_shape(joints: np.ndarray) -> np.ndarray:
    """joints를 (N, 33, 3) 형태로 통일."""
    if joints.ndim == 2 and joints.shape[1] == 99:
        return joints.reshape(-1, 33, 3)
    if joints.ndim == 3 and joints.shape[1:] == (33, 3):
        return joints
    raise ValueError(
        f"Unsupported joints shape: {joints.shape}. Expected (N,99) or (N,33,3)."
    )


def load_npz_files(npz_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """여러 npz를 로드하여 joints, betas를 concat."""
    all_joints: List[np.ndarray] = []
    all_betas: List[np.ndarray] = []

    for path in npz_paths:
        data = np.load(path, allow_pickle=True)
        if "joints" not in data or "betas" not in data:
            raise KeyError(f"{path} 에 joints/betas 키가 없습니다.")
        joints = ensure_joints_shape(data["joints"]).astype(np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32)

        if betas.ndim != 2 or betas.shape[1] != 10:
            raise ValueError(f"{path} betas shape invalid: {betas.shape}, expected (N,10)")
        if joints.shape[0] != betas.shape[0]:
            raise ValueError(
                f"{path} sample mismatch: joints={joints.shape[0]}, betas={betas.shape[0]}"
            )

        all_joints.append(joints)
        all_betas.append(betas)

    joints_cat = np.concatenate(all_joints, axis=0)
    betas_cat = np.concatenate(all_betas, axis=0)
    return joints_cat, betas_cat


def _bbox_width_height(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    points_xy: (N, K, 2)
    return width, height: (N,)
    """
    x_min = np.min(points_xy[:, :, 0], axis=1)
    x_max = np.max(points_xy[:, :, 0], axis=1)
    y_min = np.min(points_xy[:, :, 1], axis=1)
    y_max = np.max(points_xy[:, :, 1], axis=1)
    return np.maximum(x_max - x_min, 0.0), np.maximum(y_max - y_min, 0.0)


def extract_face_features(joints_33x3: np.ndarray) -> Dict[str, np.ndarray]:
    """얼굴/상체 비율 기반 1차 feature 추출."""
    face_points = joints_33x3[:, FACE_LANDMARKS, :2]  # (N,11,2)
    face_w, face_h = _bbox_width_height(face_points)
    face_area = face_w * face_h

    face_center_y = np.mean(face_points[:, :, 1], axis=1)
    face_depth_mean = np.mean(joints_33x3[:, FACE_LANDMARKS, 2], axis=1)

    torso_points = joints_33x3[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], :2]
    torso_w, torso_h = _bbox_width_height(torso_points)
    torso_area = torso_w * torso_h

    shoulder_width = np.linalg.norm(
        joints_33x3[:, LEFT_SHOULDER, :2] - joints_33x3[:, RIGHT_SHOULDER, :2], axis=1
    )

    face_to_torso_area_ratio = face_area / (torso_area + EPS)
    face_to_shoulder_ratio = face_w / (shoulder_width + EPS)

    return {
        "face_area": face_area,
        "face_width": face_w,
        "face_height": face_h,
        "face_center_y": face_center_y,
        "face_depth_mean": face_depth_mean,
        "torso_area": torso_area,
        "face_to_torso_area_ratio": face_to_torso_area_ratio,
        "shoulder_width": shoulder_width,
        "face_to_shoulder_ratio": face_to_shoulder_ratio,
    }


def valid_rows(mask_features: Dict[str, np.ndarray]) -> np.ndarray:
    """모든 feature가 finite이고 0으로 완전 실패한 샘플을 제거하기 위한 마스크 생성."""
    keys = list(mask_features.keys())
    stacked = np.stack([mask_features[k] for k in keys], axis=1)
    finite_mask = np.all(np.isfinite(stacked), axis=1)

    # 얼굴 bbox가 0이면 관측 실패 가능성이 높으므로 제외
    non_zero_mask = mask_features["face_area"] > 0
    return finite_mask & non_zero_mask


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    denom = np.sqrt(np.sum(x_c * x_c) * np.sum(y_c * y_c))
    if denom < EPS:
        return np.nan
    return float(np.sum(x_c * y_c) / denom)


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """SciPy 없이 tie-average rank 계산."""
    x = np.asarray(x, dtype=np.float64)
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))

    x_sorted = x[sorter]
    ranks = np.zeros(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x_sorted[j] == x_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = avg_rank
        i = j

    return ranks[inv]


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average_ties(x)
    ry = rankdata_average_ties(y)
    return pearson_corr(rx, ry)


def single_feature_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """y ~ a*x + b 최소제곱 R^2."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    var_x = np.sum((x - x_mean) ** 2)
    if var_x < EPS:
        return np.nan

    cov_xy = np.sum((x - x_mean) * (y - y_mean))
    a = cov_xy / var_x
    b = y_mean - a * x_mean
    y_pred = a * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot < EPS:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def analyze(features: Dict[str, np.ndarray], betas: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
    """feature별 beta(10개) 상관/설명력 분석."""
    results: Dict[str, Dict[str, List[float]]] = {}
    for feat_name, x in features.items():
        pearsons = []
        spearmans = []
        r2s = []
        for beta_idx in range(10):
            y = betas[:, beta_idx]
            pearsons.append(pearson_corr(x, y))
            spearmans.append(spearman_corr(x, y))
            r2s.append(single_feature_linear_r2(x, y))

        results[feat_name] = {
            "pearson": pearsons,
            "spearman": spearmans,
            "linear_r2": r2s,
        }
    return results


def build_markdown_report(
    data_paths: List[Path],
    sample_count_total: int,
    sample_count_used: int,
    results: Dict[str, Dict[str, List[float]]],
) -> str:
    lines: List[str] = []

    lines.append("# Face Feature vs SMPL Beta Correlation Report")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    for p in data_paths:
        lines.append(f"- {p.as_posix()}")
    lines.append(f"- total_samples: {sample_count_total}")
    lines.append(f"- used_samples_after_filter: {sample_count_used}")
    lines.append("")
    lines.append("## Metric Definition")
    lines.append("")
    lines.append("- Pearson: 선형 상관")
    lines.append("- Spearman: 단조 상관 (순위 기반)")
    lines.append("- Linear R2: 단일 feature 선형회귀 설명력")
    lines.append("")

    for feat_name, metric_dict in results.items():
        lines.append(f"## Feature: {feat_name}")
        lines.append("")
        lines.append("| beta_idx | pearson | spearman | linear_r2 |")
        lines.append("|---:|---:|---:|---:|")
        for i in range(10):
            p = metric_dict["pearson"][i]
            s = metric_dict["spearman"][i]
            r2 = metric_dict["linear_r2"][i]
            lines.append(f"| {i} | {p:.6f} | {s:.6f} | {r2:.6f} |")

        abs_pearson = [abs(v) if np.isfinite(v) else -1 for v in metric_dict["pearson"]]
        best_beta = int(np.argmax(abs_pearson))
        best_val = metric_dict["pearson"][best_beta]
        lines.append("")
        lines.append(
            f"- strongest_abs_pearson_beta: beta_{best_beta} ({best_val:.6f})"
        )
        lines.append("")

    return "\n".join(lines)


def print_console_summary(results: Dict[str, Dict[str, List[float]]]) -> None:
    print("\n" + "=" * 80)
    print("Face-derived feature correlation summary (best |Pearson| among beta_0..beta_9)")
    print("=" * 80)

    summary = []
    for feat_name, metric_dict in results.items():
        pearsons = metric_dict["pearson"]
        abs_pearsons = [abs(v) if np.isfinite(v) else -1 for v in pearsons]
        best_beta = int(np.argmax(abs_pearsons))
        best_val = pearsons[best_beta]
        summary.append((feat_name, best_beta, best_val))

    summary.sort(key=lambda x: abs(x[2]), reverse=True)
    for feat_name, beta_idx, value in summary:
        print(f"{feat_name:28s} -> beta_{beta_idx}: pearson={value: .6f}")

    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze face-area-derived features vs SMPL betas")
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="NPZ files containing joints and betas",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai_service/outputs/visual/face_beta_correlation_report.md",
        help="Output markdown report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    ai_service_root = script_path.parents[1]
    project_root = script_path.parents[2]

    raw_data_paths = [Path(p) for p in args.data]
    data_paths: List[Path] = []
    missing_raw_paths: List[Path] = []

    for raw_path in raw_data_paths:
        resolved = resolve_existing_input_path(raw_path, project_root, ai_service_root)
        if resolved is None:
            missing_raw_paths.append(raw_path)
        else:
            data_paths.append(resolved)

    if missing_raw_paths:
        missing_text = ", ".join(str(p) for p in missing_raw_paths)
        raise FileNotFoundError(
            "Data file not found: "
            f"{missing_text}\n"
            f"cwd={Path.cwd()}\n"
            "Try one of:\n"
            "1) repo root에서 실행: python ai_service/training/analyze_face_beta_correlation.py --data data/processed/train_data.npz data/processed/validation_data.npz data/processed/test_data.npz\n"
            "2) ai_service에서 실행: python training/analyze_face_beta_correlation.py --data ../data/processed/train_data.npz ../data/processed/validation_data.npz ../data/processed/test_data.npz"
        )

    joints, betas = load_npz_files(data_paths)
    features = extract_face_features(joints)

    mask = valid_rows(features)
    filtered_features = {k: v[mask] for k, v in features.items()}
    filtered_betas = betas[mask]

    if filtered_betas.shape[0] < 100:
        raise RuntimeError(
            f"Valid sample too small after filtering: {filtered_betas.shape[0]} (need >= 100)"
        )

    results = analyze(filtered_features, filtered_betas)
    print_console_summary(results)

    report = build_markdown_report(
        data_paths=data_paths,
        sample_count_total=joints.shape[0],
        sample_count_used=filtered_betas.shape[0],
        results=results,
    )

    output_path = resolve_output_path(Path(args.output), project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
