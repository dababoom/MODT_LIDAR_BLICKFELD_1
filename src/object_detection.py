from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd

import data_io
from pipeline_config import DETECTION_HYPERPARAMETERS

try:
    from point_cloud_visualization import visualize_lidar_frames_o3d
except ModuleNotFoundError:
    from src.point_cloud_visualization import visualize_lidar_frames_o3d

STATIC_CLASS_ID = 0
UNKNOWN_CLASS_ID = 1
MOVING_CLASS_ID = 2

CLASS_ID_TO_NAME = {
    STATIC_CLASS_ID: "static",
    UNKNOWN_CLASS_ID: "unknown",
    MOVING_CLASS_ID: "moving",
}
CLASS_NAME_LOOKUP = np.array(
    [
        CLASS_ID_TO_NAME[STATIC_CLASS_ID],
        CLASS_ID_TO_NAME[UNKNOWN_CLASS_ID],
        CLASS_ID_TO_NAME[MOVING_CLASS_ID],
    ],
    dtype=object,
)


@dataclass(slots=True)
class MotionDetectionConfig:
    voxel_size: float = DETECTION_HYPERPARAMETERS.voxel_size
    window_size: int = DETECTION_HYPERPARAMETERS.window_size
    moving_ratio_max: float = DETECTION_HYPERPARAMETERS.moving_ratio_max
    static_ratio_min: float = DETECTION_HYPERPARAMETERS.static_ratio_min
    dbscan_eps: float = DETECTION_HYPERPARAMETERS.dbscan_eps
    dbscan_min_points: int = DETECTION_HYPERPARAMETERS.dbscan_min_points
    min_cluster_points: int = DETECTION_HYPERPARAMETERS.min_cluster_points
    input_separator: str = ";"
    output_separator: str = ";"
    labels_output_dir: str | Path = "data/motion_labels"
    measurements_output_dir: str | Path = "data/measurements"
    clear_outputs: bool = True
    visualization: bool = False
    visualization_max_frames: int | None = 300
    visualization_frame_stride: int = 1
    visualization_window_title: str = "motion detection"
    visualization_camera_config_name: str = "wide_eagle"


@dataclass(slots=True)
class MotionDetectionResult:
    static_indices_by_frame: dict[int, list[int]]
    unknown_indices_by_frame: dict[int, list[int]]
    moving_indices_by_frame: dict[int, list[int]]
    moving_cluster_indices_by_frame: dict[int, dict[int, list[int]]]
    label_files: list[Path]
    measurement_files: list[Path]


def motion_detection(path: str | Path = "data/preprocessed", *, config: MotionDetectionConfig | None = None, max_files: int | None = None) -> MotionDetectionResult:
    """Run the motion-based detection stage on preprocessed LiDAR frames."""
    cfg = config if config is not None else MotionDetectionConfig()
    _validate_config(cfg)

    input_dir = Path(path)
    frame_files = list_preprocessed_frames(input_dir, max_files=max_files)

    labels_output_dir = Path(cfg.labels_output_dir)
    measurements_output_dir = Path(cfg.measurements_output_dir)
    _prepare_output_directory(labels_output_dir, clear_outputs=cfg.clear_outputs)
    _prepare_output_directory(measurements_output_dir, clear_outputs=cfg.clear_outputs)

    history: deque[set[tuple[int, int, int]]] = deque(maxlen=cfg.window_size)
    occupancy_counts: Counter[tuple[int, int, int]] = Counter()

    static_indices_by_frame: dict[int, list[int]] = {}
    unknown_indices_by_frame: dict[int, list[int]] = {}
    moving_indices_by_frame: dict[int, list[int]] = {}
    moving_cluster_indices_by_frame: dict[int, dict[int, list[int]]] = {}
    label_files: list[Path] = []
    measurement_files: list[Path] = []
    visualization_frames: list[pd.DataFrame] = []
    measurement_bounding_boxes: list[dict[str, Any]] = []

    for frame_file in frame_files:
        frame_id, frame_table = _load_frame_table(frame_file, separator=cfg.input_separator)
        points_xyz = frame_table.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
        point_count = int(points_xyz.shape[0])

        class_ids = np.full(point_count, UNKNOWN_CLASS_ID, dtype=np.int8)
        voxel_ratios = np.full(point_count, np.nan, dtype=np.float32)
        cluster_ids = np.full(point_count, -1, dtype=np.int32)
        voxel_ids = np.zeros((point_count, 3), dtype=np.int32)
        current_voxel_set: set[tuple[int, int, int]] = set()

        if point_count > 0:
            voxel_ids = _points_to_voxel_ids(points_xyz, voxel_size=cfg.voxel_size)
            unique_voxel_ids, inverse_indices = np.unique(voxel_ids, axis=0, return_inverse=True)
            current_voxel_set = {tuple(int(value) for value in voxel) for voxel in unique_voxel_ids}

            history_size = len(history)
            if history_size > 0:
                voxel_ratios_unique = np.array(
                    [occupancy_counts.get(tuple(int(value) for value in voxel), 0) / history_size for voxel in unique_voxel_ids],
                    dtype=np.float32,
                )
                voxel_ratios = voxel_ratios_unique[inverse_indices]

                class_ids[voxel_ratios <= cfg.moving_ratio_max] = MOVING_CLASS_ID
                class_ids[voxel_ratios >= cfg.static_ratio_min] = STATIC_CLASS_ID

                moving_indices = np.flatnonzero(class_ids == MOVING_CLASS_ID)
                cluster_ids, cluster_indices = _cluster_moving_points(points_xyz=points_xyz, moving_indices=moving_indices, dbscan_eps=cfg.dbscan_eps, dbscan_min_points=cfg.dbscan_min_points, min_cluster_points=cfg.min_cluster_points)
            else:
                cluster_indices = {}
        else:
            cluster_indices = {}

        static_indices_by_frame[frame_id] = np.flatnonzero(class_ids == STATIC_CLASS_ID).astype(np.int64).tolist()
        unknown_indices_by_frame[frame_id] = np.flatnonzero(class_ids == UNKNOWN_CLASS_ID).astype(np.int64).tolist()
        moving_indices_by_frame[frame_id] = np.flatnonzero(class_ids == MOVING_CLASS_ID).astype(np.int64).tolist()
        moving_cluster_indices_by_frame[frame_id] = cluster_indices

        motion_class_names = CLASS_NAME_LOOKUP[class_ids] if point_count > 0 else np.array([], dtype=object)
        labeled_frame = frame_table.copy()
        labeled_frame["voxel_ix"] = voxel_ids[:, 0]
        labeled_frame["voxel_iy"] = voxel_ids[:, 1]
        labeled_frame["voxel_iz"] = voxel_ids[:, 2]
        labeled_frame["voxel_ratio"] = voxel_ratios
        labeled_frame["motion_class_id"] = class_ids.astype(np.int8)
        labeled_frame["motion_class"] = motion_class_names
        labeled_frame["motion_cluster_id"] = cluster_ids

        label_file = labels_output_dir / f"frame-{frame_id}.csv"
        labeled_frame.to_csv(label_file, sep=cfg.output_separator, index=False)
        label_files.append(label_file)

        frame_measurements = _build_frame_measurements(frame_id=frame_id, points_xyz=points_xyz, intensities=frame_table["intensity"].to_numpy(dtype=np.float32) if "intensity" in frame_table.columns else None, cluster_indices=cluster_indices)
        measurement_file = measurements_output_dir / f"frame-{frame_id}.csv"
        frame_measurements.to_csv(measurement_file, sep=cfg.output_separator, index=False)
        measurement_files.append(measurement_file)

        if cfg.visualization:
            visualization_frames.append(frame_table)
            measurement_bounding_boxes.extend(
                _build_measurement_bounding_boxes(frame_measurements, box_color=(0.2, 1.0, 0.2))
            )

        _update_rolling_occupancy(history=history, occupancy_counts=occupancy_counts, current_voxel_set=current_voxel_set)

    print(f"[motion_detection] Processed {len(frame_files)} frames. Labeled frame files: {labels_output_dir}. Measurements: {measurements_output_dir}.")
    if cfg.visualization and visualization_frames:
        frames_for_visualization = pd.concat(visualization_frames, ignore_index=True)
        selected_frame_ids = sorted(frames_for_visualization["frame_id"].unique().tolist())[:: cfg.visualization_frame_stride]
        if cfg.visualization_max_frames is not None:
            selected_frame_ids = selected_frame_ids[: cfg.visualization_max_frames]
        selected_frame_id_set = set(selected_frame_ids)
        measurement_bounding_boxes_for_selected_frames = [
            box for box in measurement_bounding_boxes if int(box["frame_id"]) in selected_frame_id_set
        ]
        visualize_lidar_frames_o3d(
            frames_for_visualization,
            color_mode="single",
            point_color=(0.6, 0.6, 0.6),
            flagged_indices_by_frame=[unknown_indices_by_frame, moving_indices_by_frame],
            flagged_color=[(1.0, 0.65, 0.0), (1.0, 0.0, 0.0)],
            max_frames=cfg.visualization_max_frames,
            frame_stride=cfg.visualization_frame_stride,
            exclude_flagged_from_frames=True,
            camera_config_name=cfg.visualization_camera_config_name,
            window_title=cfg.visualization_window_title,
        )

        visualize_lidar_frames_o3d(
            frames_for_visualization,
            color_mode="single",
            point_color=(0.95, 0.95, 0.95),
            flagged_indices_by_frame=moving_indices_by_frame,
            flagged_color=(1.0, 0.0, 0.0),
            max_frames=cfg.visualization_max_frames,
            frame_stride=cfg.visualization_frame_stride,
            exclude_flagged_from_frames=True,
            bounding_boxes=measurement_bounding_boxes_for_selected_frames,
            show_box_names=False,
            default_box_color=(0.2, 1.0, 0.2),
            camera_config_name=cfg.visualization_camera_config_name,
            window_title=f"{cfg.visualization_window_title} (moving vs non-moving + boxes)",
        )

    return MotionDetectionResult(static_indices_by_frame=static_indices_by_frame, unknown_indices_by_frame=unknown_indices_by_frame, moving_indices_by_frame=moving_indices_by_frame, moving_cluster_indices_by_frame=moving_cluster_indices_by_frame, label_files=label_files, measurement_files=measurement_files)


def list_preprocessed_frames(preprocessed_dir: str | Path, max_files: int | None = None) -> list[Path]:
    """Return canonical preprocessed frame files for the detection stage."""
    input_dir = Path(preprocessed_dir)
    files = data_io.list_saved_frame_files(input_dir, max_files=max_files)
    if not files:
        raise FileNotFoundError(f"No preprocessed frame CSV files found in: {input_dir}")
    return files


def _load_frame_table(frame_file: Path, *, separator: str) -> tuple[int, pd.DataFrame]:
    """Load one canonical preprocessed frame table for detection."""
    return data_io.load_saved_frame_table(frame_file, separator=separator)


def _points_to_voxel_ids(points_xyz: np.ndarray, *, voxel_size: float) -> np.ndarray:
    """Map XYZ coordinates to integer voxel indices."""
    return np.floor(points_xyz / voxel_size).astype(np.int32)


def _cluster_moving_points(*, points_xyz: np.ndarray, moving_indices: np.ndarray, dbscan_eps: float, dbscan_min_points: int, min_cluster_points: int) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Cluster moving points with DBSCAN and remap cluster ids to dense integers."""
    point_count = int(points_xyz.shape[0])
    cluster_ids = np.full(point_count, -1, dtype=np.int32)
    cluster_indices: dict[int, list[int]] = {}

    if moving_indices.size == 0:
        return cluster_ids, cluster_indices

    moving_xyz = points_xyz[moving_indices].astype(np.float64, copy=False)
    moving_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(moving_xyz))
    dbscan_labels = np.asarray(moving_cloud.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False), dtype=np.int32)

    valid_mask = dbscan_labels >= 0
    if not valid_mask.any():
        return cluster_ids, cluster_indices

    cluster_labels, cluster_sizes = np.unique(dbscan_labels[valid_mask], return_counts=True)
    kept_labels = cluster_labels[cluster_sizes >= min_cluster_points]
    remapped_labels = {int(old_label): int(new_label) for new_label, old_label in enumerate(sorted(kept_labels.tolist()))}

    for local_idx, raw_label in enumerate(dbscan_labels):
        new_label = remapped_labels.get(int(raw_label))
        if new_label is None:
            continue
        global_idx = int(moving_indices[local_idx])
        cluster_ids[global_idx] = new_label

    kept_cluster_ids = np.unique(cluster_ids[cluster_ids >= 0])
    for cluster_id in kept_cluster_ids:
        cluster_indices[int(cluster_id)] = np.flatnonzero(cluster_ids == int(cluster_id)).astype(np.int64).tolist()

    return cluster_ids, cluster_indices


def _build_frame_measurements(*, frame_id: int, points_xyz: np.ndarray, intensities: np.ndarray | None, cluster_indices: dict[int, list[int]]) -> pd.DataFrame:
    """Summarize each moving cluster as one measurement row."""
    measurement_rows: list[dict[str, float | int]] = []

    for cluster_id, point_indices in sorted(cluster_indices.items()):
        cluster_idx = np.asarray(point_indices, dtype=np.int64)
        cluster_points = points_xyz[cluster_idx]
        cluster_size = int(cluster_points.shape[0])
        if cluster_size == 0:
            continue

        centroid = cluster_points.mean(axis=0)
        minimum = cluster_points.min(axis=0)
        maximum = cluster_points.max(axis=0)
        extent = maximum - minimum
        covariance = np.cov(cluster_points.T, bias=False) if cluster_size >= 2 else np.zeros((3, 3), dtype=np.float64)

        row: dict[str, float | int] = {
            "frame_id": int(frame_id),
            "cluster_id": int(cluster_id),
            "point_count": cluster_size,
            "bbox_volume": float(np.prod(extent)),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "bbox_min_x": float(minimum[0]),
            "bbox_min_y": float(minimum[1]),
            "bbox_min_z": float(minimum[2]),
            "bbox_max_x": float(maximum[0]),
            "bbox_max_y": float(maximum[1]),
            "bbox_max_z": float(maximum[2]),
            "bbox_dx": float(extent[0]),
            "bbox_dy": float(extent[1]),
            "bbox_dz": float(extent[2]),
            "cov_xx": float(covariance[0, 0]),
            "cov_xy": float(covariance[0, 1]),
            "cov_xz": float(covariance[0, 2]),
            "cov_yx": float(covariance[1, 0]),
            "cov_yy": float(covariance[1, 1]),
            "cov_yz": float(covariance[1, 2]),
            "cov_zx": float(covariance[2, 0]),
            "cov_zy": float(covariance[2, 1]),
            "cov_zz": float(covariance[2, 2]),
        }
        if intensities is not None and intensities.size > 0:
            row["mean_intensity"] = float(np.nanmean(intensities[cluster_idx]))

        measurement_rows.append(row)

    if not measurement_rows:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "cluster_id",
                "point_count",
                "bbox_volume",
                "centroid_x",
                "centroid_y",
                "centroid_z",
                "bbox_min_x",
                "bbox_min_y",
                "bbox_min_z",
                "bbox_max_x",
                "bbox_max_y",
                "bbox_max_z",
                "bbox_dx",
                "bbox_dy",
                "bbox_dz",
                "cov_xx",
                "cov_xy",
                "cov_xz",
                "cov_yx",
                "cov_yy",
                "cov_yz",
                "cov_zx",
                "cov_zy",
                "cov_zz",
                "mean_intensity",
            ]
        )

    return pd.DataFrame(measurement_rows)


def _build_measurement_bounding_boxes(
    frame_measurements: pd.DataFrame,
    *,
    box_color: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Convert per-cluster measurements into visualization boxes."""
    if frame_measurements.empty:
        return []

    required_columns = (
        "frame_id",
        "cluster_id",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "bbox_dx",
        "bbox_dy",
        "bbox_dz",
    )
    missing_columns = [column_name for column_name in required_columns if column_name not in frame_measurements.columns]
    if missing_columns:
        raise ValueError(f"Measurement table is missing required columns for bounding boxes: {missing_columns}")

    boxes: list[dict[str, Any]] = []
    for row in frame_measurements.itertuples(index=False):
        extent = np.asarray([row.bbox_dx, row.bbox_dy, row.bbox_dz], dtype=np.float32)
        if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
            continue

        boxes.append(
            {
                "frame_id": int(row.frame_id),
                "name": f"cluster_{int(row.cluster_id)}",
                "center": (float(row.centroid_x), float(row.centroid_y), float(row.centroid_z)),
                "extent": (float(row.bbox_dx), float(row.bbox_dy), float(row.bbox_dz)),
                "color": box_color,
            }
        )

    return boxes


def _update_rolling_occupancy(
    *,
    history: deque[set[tuple[int, int, int]]],
    occupancy_counts: Counter[tuple[int, int, int]],
    current_voxel_set: set[tuple[int, int, int]],
) -> None:
    """Update the voxel occupancy window used for motion classification."""
    if history.maxlen is not None and len(history) == history.maxlen and len(history) > 0:
        oldest_voxel_set = history.popleft()
        for voxel_key in oldest_voxel_set:
            occupancy_counts[voxel_key] -= 1
            if occupancy_counts[voxel_key] <= 0:
                del occupancy_counts[voxel_key]

    history.append(current_voxel_set)
    for voxel_key in current_voxel_set:
        occupancy_counts[voxel_key] += 1


def _prepare_output_directory(output_dir: Path, *, clear_outputs: bool) -> None:
    """Create an output directory and optionally clear prior frame files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not clear_outputs:
        return
    for existing_file in output_dir.glob("frame-*.csv"):
        existing_file.unlink()


def _validate_config(config: MotionDetectionConfig) -> None:
    """Validate the motion-detection hyperparameters before processing."""
    if config.voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    if config.window_size <= 0:
        raise ValueError("window_size must be > 0.")
    if not 0 <= config.moving_ratio_max <= 1:
        raise ValueError("moving_ratio_max must be in [0, 1].")
    if not 0 <= config.static_ratio_min <= 1:
        raise ValueError("static_ratio_min must be in [0, 1].")
    if config.moving_ratio_max >= config.static_ratio_min:
        raise ValueError("moving_ratio_max must be < static_ratio_min.")
    if config.dbscan_eps <= 0:
        raise ValueError("dbscan_eps must be > 0.")
    if config.dbscan_min_points <= 0:
        raise ValueError("dbscan_min_points must be > 0.")
    if config.min_cluster_points <= 0:
        raise ValueError("min_cluster_points must be > 0.")
    if config.visualization_frame_stride <= 0:
        raise ValueError("visualization_frame_stride must be > 0.")
