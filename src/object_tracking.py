from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stonesoup.dataassociator.probability import JPDA
from stonesoup.functions import gm_reduce_single
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import KalmanUpdater

import data_io
from pipeline_config import TRACKING_HYPERPARAMETERS


TRACK_POSTERIOR_COVARIANCE_COLUMNS = ("cov_x", "cov_y", "cov_z")
TRACK_ASSOCIATED_MEASUREMENT_COLUMNS = ("measurement_centroid_x", "measurement_centroid_y", "measurement_centroid_z")
TRACK_BBOX_COLUMNS = ("bbox_mean_center_x", "bbox_mean_center_y", "bbox_mean_center_z", "bbox_dx", "bbox_dy", "bbox_dz")
TRACK_BBOX_SIZE_COLUMNS = TRACK_BBOX_COLUMNS[3:]
TRACK_MEASUREMENT_COLUMNS = ("point_count", "bbox_volume", "mean_intensity")
TRACK_CLASSIFICATION_COLUMN = "classification_label"
TRACK_STATE_COLUMNS = ("x", "y", "z", *TRACK_BBOX_SIZE_COLUMNS)
TRACK_OUTPUT_COLUMNS = [
    "frame_id",
    "track_id",
    "x",
    "vx",
    "y",
    "vy",
    "z",
    "vz",
    "speed_mps",
    "missed_frames",
    *TRACK_POSTERIOR_COVARIANCE_COLUMNS,
    *TRACK_ASSOCIATED_MEASUREMENT_COLUMNS,
    *TRACK_BBOX_COLUMNS,
    *TRACK_MEASUREMENT_COLUMNS,
    TRACK_CLASSIFICATION_COLUMN,
]
CLASSIFICATION_NONE = "unclassified"
CLASSIFICATION_PEDESTRIAN = "pedestrian"
CLASSIFICATION_CYCLIST = "cyclist"
CLASSIFICATION_VEHICLE = "vehicle"
CLASSIFICATION_RANK = {
    CLASSIFICATION_NONE: 0,
    CLASSIFICATION_PEDESTRIAN: 1,
    CLASSIFICATION_CYCLIST: 2,
    CLASSIFICATION_VEHICLE: 3,
}
MEASUREMENT_BOX_COLOR = (1.0, 0.55, 0.0)
STATE_BOX_COLOR = (0.1, 0.85, 0.1)


@dataclass(slots=True)
class JPDATrackingConfig:
    """Configuration for a compact JPDA pipeline from preprocessed frames to tracked states."""
    preprocessed_dir: str | Path = "data/preprocessed"
    labels_output_dir: str | Path = "data/motion_labels"
    measurements_dir: str | Path = "data/measurements"
    tracks_output_file: str | Path = "data/tracks/tracks.csv"
    input_separator: str = ";"
    output_separator: str = ";"
    dt_seconds: float = TRACKING_HYPERPARAMETERS.dt_seconds
    max_tracks: int = TRACKING_HYPERPARAMETERS.max_tracks
    motion_sigma: float = TRACKING_HYPERPARAMETERS.motion_sigma
    measurement_sigma: float = TRACKING_HYPERPARAMETERS.measurement_sigma
    init_pos_sigma: float = TRACKING_HYPERPARAMETERS.init_pos_sigma
    init_vel_sigma: float = TRACKING_HYPERPARAMETERS.init_vel_sigma
    prob_detect: float = TRACKING_HYPERPARAMETERS.prob_detect
    prob_gate: float = TRACKING_HYPERPARAMETERS.prob_gate
    clutter_density: float = TRACKING_HYPERPARAMETERS.clutter_density
    birth_distance_m: float = TRACKING_HYPERPARAMETERS.birth_distance_m
    miss_probability_threshold: float = TRACKING_HYPERPARAMETERS.miss_probability_threshold
    max_missed: int = TRACKING_HYPERPARAMETERS.max_missed
    min_hits_confirm: int = TRACKING_HYPERPARAMETERS.min_hits_confirm
    max_files: int | None = None
    visualization: bool = False
    visualization_max_frames: int | None = 3000
    visualization_frame_stride: int = 1
    visualization_window_title: str = "JPDA tracking (3D)"
    visualization_camera_config_name: str = "wide_eagle"
    visualization_point_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visualization_box_line_width: float = 2.0
    visualization_show_box_names: bool = True


@dataclass(slots=True)
class _TrackingRuntime:
    """Mutable runtime state for one JPDA tracking run."""
    tracks: set[Track] = field(default_factory=set)
    track_ids: dict[Track, int] = field(default_factory=dict)
    missed: dict[int, int] = field(default_factory=dict)
    hits: dict[int, int] = field(default_factory=dict)
    confirmed_track_ids: set[int] = field(default_factory=set)
    confirmed_output_track_ids: dict[int, int] = field(default_factory=dict)
    last_track_box_sizes: dict[int, dict[str, float]] = field(default_factory=dict)
    track_classifications: dict[int, str] = field(default_factory=dict)
    next_track_id: int = 1
    next_confirmed_output_track_id: int = 1


def run_tracking_jpda(*, config: JPDATrackingConfig | None = None) -> Path:
    """Run JPDA tracking directly from saved measurement frames to tracks_jpda.csv."""
    cfg = config or JPDATrackingConfig()
    if cfg.min_hits_confirm <= 0:
        raise ValueError("min_hits_confirm must be > 0.")

    measurement_files = data_io.list_saved_frame_files(cfg.measurements_dir, max_files=cfg.max_files)
    processed_frame_ids = [data_io.frame_id_from_filename(path, strict=True) for path in measurement_files]
    measurement_model, updater, associator = _build_tracking_components(cfg)
    runtime = _TrackingRuntime()
    rows: list[dict[str, float | int | str]] = []
    base_time = datetime(2020, 11, 25, 20, 1, 45)

    for step, frame_file in enumerate(measurement_files):
        frame_id = data_io.frame_id_from_filename(frame_file, strict=True)
        timestamp = base_time + timedelta(seconds=cfg.dt_seconds * step)
        frame_track_boxes: dict[int, dict[str, float | int]] = {}
        table = pd.read_csv(frame_file, sep=cfg.input_separator)
        detections = _build_detections(table=table, timestamp=timestamp, measurement_model=measurement_model)
        _update_existing_tracks(
            runtime=runtime,
            associator=associator,
            updater=updater,
            detections=detections,
            timestamp=timestamp,
            config=cfg,
            frame_track_boxes=frame_track_boxes,
        )
        _birth_new_tracks(
            runtime=runtime,
            detections=detections,
            timestamp=timestamp,
            config=cfg,
            frame_track_boxes=frame_track_boxes,
        )
        _prune_stale_tracks(runtime=runtime, max_missed=cfg.max_missed)
        rows.extend(
            _collect_confirmed_track_rows(
                runtime=runtime,
                frame_id=frame_id,
                frame_track_boxes=frame_track_boxes,
            )
        )

    output_path = Path(cfg.tracks_output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=TRACK_OUTPUT_COLUMNS).to_csv(output_path, sep=cfg.output_separator, index=False)
    print(
        f"[tracking_jpda] Frames={len(measurement_files)} "
        f"TracksCreated={runtime.next_track_id - 1} "
        f"ConfirmedEver={len(runtime.confirmed_output_track_ids)} Output={output_path}"
    )

    if cfg.visualization:
        _visualize_jpda_tracking(rows=rows, config=cfg, processed_frame_ids=processed_frame_ids)

    return output_path


def _build_tracking_components(cfg: JPDATrackingConfig) -> tuple[LinearGaussian, KalmanUpdater, JPDA]:
    """Create the Stone Soup models used by the JPDA tracker."""
    transition_model = CombinedLinearGaussianTransitionModel(
        (
            ConstantVelocity(cfg.motion_sigma),
            ConstantVelocity(cfg.motion_sigma),
            ConstantVelocity(cfg.motion_sigma),
        )
    )
    predictor = KalmanPredictor(transition_model)
    measurement_model = LinearGaussian(
        ndim_state=6,
        mapping=[0, 2, 4],
        noise_covar=np.diag(
            [
                cfg.measurement_sigma ** 2,
                cfg.measurement_sigma ** 2,
                cfg.measurement_sigma ** 2,
            ]
        ),
    )
    updater = KalmanUpdater(measurement_model)
    hypothesiser = PDAHypothesiser(
        predictor=predictor,
        updater=updater,
        clutter_spatial_density=cfg.clutter_density,
        prob_detect=cfg.prob_detect,
        prob_gate=cfg.prob_gate,
    )
    return measurement_model, updater, JPDA(hypothesiser=hypothesiser)


def _build_detections(
    *,
    table: pd.DataFrame,
    timestamp: datetime,
    measurement_model: LinearGaussian,
) -> set[Detection]:
    """Convert one measurement table into Stone Soup detections."""
    return {
        Detection(
            np.array(
                [
                    [float(row.centroid_x)],
                    [float(row.centroid_y)],
                    [float(row.centroid_z)],
                ],
                dtype=np.float64,
            ),
            timestamp=timestamp,
            measurement_model=measurement_model,
            metadata=_measurement_bbox_metadata(row),
        )
        for row in table.itertuples(index=False)
    }


def _update_existing_tracks(
    *,
    runtime: _TrackingRuntime,
    associator: JPDA,
    updater: KalmanUpdater,
    detections: set[Detection],
    timestamp: datetime,
    config: JPDATrackingConfig,
    frame_track_boxes: dict[int, dict[str, float | int]],
) -> None:
    """Associate detections to existing tracks and update their posterior state."""
    if not runtime.tracks:
        return

    hypotheses = associator.associate(runtime.tracks, detections, timestamp)
    for track in list(runtime.tracks):
        multi_hypothesis = hypotheses[track]
        _append_reduced_track_update(track=track, multi_hypothesis=multi_hypothesis, updater=updater, timestamp=timestamp)

        track_id = runtime.track_ids[track]
        best_detection_hypothesis = max(
            (hypothesis for hypothesis in multi_hypothesis if hypothesis),
            key=lambda hypothesis: float(hypothesis.probability),
            default=None,
        )
        best_detection_prob = (
            float(best_detection_hypothesis.probability)
            if best_detection_hypothesis is not None
            else 0.0
        )
        detection_supported = best_detection_prob >= config.miss_probability_threshold
        runtime.missed[track_id] = 0 if detection_supported else runtime.missed[track_id] + 1
        if not detection_supported:
            continue

        measurement_metadata = dict(getattr(best_detection_hypothesis.measurement, "metadata", {}) or {})
        _store_measurement_metadata(
            runtime=runtime,
            track_id=track_id,
            measurement_metadata=measurement_metadata,
            frame_track_boxes=frame_track_boxes,
        )
        runtime.hits[track_id] = runtime.hits.get(track_id, 1) + 1
        if runtime.hits[track_id] >= config.min_hits_confirm:
            _confirm_track(runtime=runtime, track_id=track_id)
        if measurement_metadata and track_id in runtime.confirmed_track_ids:
            runtime.track_classifications[track_id] = object_classification(
                current_label=runtime.track_classifications.get(track_id),
                x=float(track.state.state_vector[0, 0]),
                y=float(track.state.state_vector[2, 0]),
                z=float(track.state.state_vector[4, 0]),
                metadata=measurement_metadata,
            )


def _append_reduced_track_update(
    *,
    track: Track,
    multi_hypothesis: Any,
    updater: KalmanUpdater,
    timestamp: datetime,
) -> None:
    """Reduce a JPDA multi-hypothesis into one Gaussian posterior state."""
    states = [updater.update(hypothesis) if hypothesis else hypothesis.prediction for hypothesis in multi_hypothesis]
    weights = np.asarray([float(hypothesis.probability) for hypothesis in multi_hypothesis], dtype=np.float64)
    means = StateVectors([state.state_vector for state in states])
    covars = np.stack([state.covar for state in states], axis=2)
    mean, covar = gm_reduce_single(means, covars, weights)
    track.append(GaussianStateUpdate(mean, covar, multi_hypothesis, timestamp))


def _store_measurement_metadata(
    *,
    runtime: _TrackingRuntime,
    track_id: int,
    measurement_metadata: dict[str, float | int],
    frame_track_boxes: dict[int, dict[str, float | int]],
) -> None:
    """Cache the latest bounding-box metadata for a track."""
    if not measurement_metadata:
        return
    frame_track_boxes[track_id] = measurement_metadata
    runtime.last_track_box_sizes[track_id] = {
        column: float(measurement_metadata[column])
        for column in TRACK_BBOX_SIZE_COLUMNS
    }


def _birth_new_tracks(
    *,
    runtime: _TrackingRuntime,
    detections: set[Detection],
    timestamp: datetime,
    config: JPDATrackingConfig,
    frame_track_boxes: dict[int, dict[str, float | int]],
) -> None:
    """Create new tracks from detections that are far from existing tracks."""
    positions = _track_positions(runtime.tracks)
    for detection in detections:
        if len(runtime.tracks) >= config.max_tracks:
            break

        detection_position = np.array(
            [
                float(detection.state_vector[0, 0]),
                float(detection.state_vector[1, 0]),
                float(detection.state_vector[2, 0]),
            ],
            dtype=np.float64,
        )
        if (
            positions.size
            and float(np.min(np.linalg.norm(positions - detection_position, axis=1)))
            < config.birth_distance_m
        ):
            continue

        init_covar = np.diag(
            [
                config.init_pos_sigma ** 2,
                config.init_vel_sigma ** 2,
                config.init_pos_sigma ** 2,
                config.init_vel_sigma ** 2,
                config.init_pos_sigma ** 2,
                config.init_vel_sigma ** 2,
            ]
        ).astype(np.float64)
        state = GaussianState(
            np.array(
                [
                    [detection_position[0]],
                    [0.0],
                    [detection_position[1]],
                    [0.0],
                    [detection_position[2]],
                    [0.0],
                ],
                dtype=np.float64,
            ),
            init_covar,
            timestamp=timestamp,
        )
        track = Track([state])
        track_id = runtime.next_track_id
        runtime.tracks.add(track)
        runtime.track_ids[track] = track_id
        runtime.missed[track_id] = 0
        runtime.hits[track_id] = 1

        detection_metadata = dict(detection.metadata or {})
        _store_measurement_metadata(
            runtime=runtime,
            track_id=track_id,
            measurement_metadata=detection_metadata,
            frame_track_boxes=frame_track_boxes,
        )
        if config.min_hits_confirm <= 1:
            _confirm_track(runtime=runtime, track_id=track_id)
            if detection_metadata:
                runtime.track_classifications[track_id] = object_classification(
                    current_label=runtime.track_classifications.get(track_id),
                    x=float(state.state_vector[0, 0]),
                    y=float(state.state_vector[2, 0]),
                    z=float(state.state_vector[4, 0]),
                    metadata=detection_metadata,
                )

        runtime.next_track_id += 1
        positions = (
            np.vstack([positions, detection_position])
            if positions.size
            else np.array([detection_position], dtype=np.float64)
        )


def _track_positions(tracks: set[Track]) -> np.ndarray:
    """Return the current XYZ positions of all active tracks."""
    if not tracks:
        return np.empty((0, 3), dtype=np.float64)
    return np.array(
        [
            [
                float(track.state.state_vector[0, 0]),
                float(track.state.state_vector[2, 0]),
                float(track.state.state_vector[4, 0]),
            ]
            for track in tracks
        ],
        dtype=np.float64,
    )


def _confirm_track(*, runtime: _TrackingRuntime, track_id: int) -> None:
    """Assign a stable output id the first time a track becomes confirmed."""
    runtime.confirmed_track_ids.add(track_id)
    if track_id in runtime.confirmed_output_track_ids:
        return
    runtime.confirmed_output_track_ids[track_id] = runtime.next_confirmed_output_track_id
    runtime.next_confirmed_output_track_id += 1


def _prune_stale_tracks(*, runtime: _TrackingRuntime, max_missed: int) -> None:
    """Remove tracks that exceeded the allowed number of consecutive misses."""
    for track in list(runtime.tracks):
        track_id = runtime.track_ids[track]
        if runtime.missed[track_id] <= max_missed:
            continue
        runtime.tracks.remove(track)
        del runtime.track_ids[track]
        del runtime.missed[track_id]
        runtime.hits.pop(track_id, None)
        runtime.confirmed_track_ids.discard(track_id)
        runtime.last_track_box_sizes.pop(track_id, None)
        runtime.track_classifications.pop(track_id, None)


def _collect_confirmed_track_rows(
    *,
    runtime: _TrackingRuntime,
    frame_id: int,
    frame_track_boxes: dict[int, dict[str, float | int]],
) -> list[dict[str, float | int | str]]:
    """Build one output row per confirmed active track for the current frame."""
    rows: list[dict[str, float | int | str]] = []
    for track in runtime.tracks:
        state = track.state.state_vector
        state_covar = track.state.covar
        track_id = runtime.track_ids[track]
        output_track_id = runtime.confirmed_output_track_ids.get(track_id)
        if output_track_id is None:
            continue

        row = {
            "frame_id": int(frame_id),
            "track_id": int(output_track_id),
            "x": float(state[0, 0]),
            "vx": float(state[1, 0]),
            "y": float(state[2, 0]),
            "vy": float(state[3, 0]),
            "z": float(state[4, 0]),
            "vz": float(state[5, 0]),
            "speed_mps": float(np.sqrt(state[1, 0] ** 2 + state[3, 0] ** 2 + state[5, 0] ** 2)),
            "missed_frames": int(runtime.missed[track_id]),
            "cov_x": float(state_covar[0, 0]),
            "cov_y": float(state_covar[2, 2]),
            "cov_z": float(state_covar[4, 4]),
            TRACK_CLASSIFICATION_COLUMN: runtime.track_classifications.get(track_id, CLASSIFICATION_NONE),
        }
        measurement_metadata = frame_track_boxes.get(track_id)
        if measurement_metadata is not None:
            row.update(measurement_metadata)
        elif track_id in runtime.last_track_box_sizes:
            row.update(runtime.last_track_box_sizes[track_id])
        rows.append(row)
    return rows


def _visualize_jpda_tracking(
    *,
    rows: list[dict[str, float | int | str]],
    config: JPDATrackingConfig,
    processed_frame_ids: list[int] | None = None,
) -> None:
    """Render point clouds, measurement boxes, and confirmed tracks together."""
    if config.visualization_frame_stride <= 0:
        raise ValueError("visualization_frame_stride must be > 0.")
    if config.visualization_max_frames is not None and config.visualization_max_frames <= 0:
        raise ValueError("visualization_max_frames must be > 0 when provided.")
    if config.visualization_box_line_width <= 0:
        raise ValueError("visualization_box_line_width must be > 0.")

    try:
        try:
            from point_cloud_visualization import visualize_lidar_frames_o3d
        except ModuleNotFoundError:
            from src.point_cloud_visualization import visualize_lidar_frames_o3d
    except Exception as exc:
        print(f"[tracking_jpda] Visualization skipped: unable to import point cloud visualizer ({exc}).")
        return

    track_table = pd.DataFrame(rows)
    if processed_frame_ids is not None and len(processed_frame_ids) > 0:
        base_frame_ids = sorted(set(int(frame_id) for frame_id in processed_frame_ids))
    elif not track_table.empty:
        base_frame_ids = sorted(track_table["frame_id"].astype(np.int64).unique().tolist())
    else:
        print("[tracking_jpda] Visualization skipped: no frames available.")
        return

    selected_frame_ids = base_frame_ids[:: config.visualization_frame_stride]
    if config.visualization_max_frames is not None:
        selected_frame_ids = selected_frame_ids[: config.visualization_max_frames]
    if not selected_frame_ids:
        print("[tracking_jpda] Visualization skipped: selected frame range is empty.")
        return

    selected_frame_set = set(int(frame_id) for frame_id in selected_frame_ids)
    if not track_table.empty:
        track_table = track_table.loc[track_table["frame_id"].astype(np.int64).isin(selected_frame_set)].copy()

    frames = _load_preprocessed_frames(
        preprocessed_dir=config.preprocessed_dir,
        frame_ids=selected_frame_ids,
        separator=config.input_separator,
    )
    if frames.empty:
        print(f"[tracking_jpda] Visualization skipped: no preprocessed frames found in {config.preprocessed_dir}.")
        return

    measurement_boxes = _load_measurement_bounding_boxes(
        measurements_dir=config.measurements_dir,
        frame_ids=selected_frame_ids,
        separator=config.input_separator,
    )

    if track_table.empty:
        track_boxes: list[dict[str, Any]] = []
        if not measurement_boxes:
            print("[tracking_jpda] No confirmed tracks or measurement boxes in selected frames; rendering point cloud only.")
    else:
        track_boxes = _build_track_state_bounding_boxes(track_table=track_table)

    bounding_boxes = measurement_boxes + track_boxes

    visualize_lidar_frames_o3d(
        frames,
        frame_ids=selected_frame_ids,
        point_color=(0.95, 0.95, 0.95),
        max_frames=None,
        frame_stride=1,
        color_mode="single",
        bounding_boxes=bounding_boxes,
        show_box_names=config.visualization_show_box_names,
        sync_labels_with_frames=True,
        box_line_width=3.0,
        window_title=config.visualization_window_title,
        background_rgba=(0.45, 0.48, 0.52, 1.0),
    )


def _load_preprocessed_frames(*, preprocessed_dir: str | Path, frame_ids: list[int], separator: str) -> pd.DataFrame:
    """Load the selected preprocessed frames in the canonical table format."""
    return data_io.load_saved_frame_tables(
        preprocessed_dir,
        frame_ids=frame_ids,
        separator=separator,
    )


def _load_measurement_bounding_boxes(*, measurements_dir: str | Path, frame_ids: list[int], separator: str) -> list[dict[str, Any]]:
    """Load measurement boxes for the selected visualization frames."""
    directory = Path(measurements_dir)
    if not directory.exists():
        return []

    frame_id_set = {int(frame_id) for frame_id in frame_ids}
    selected_files = [
        file_path
        for file_path in data_io.list_saved_frame_files(directory)
        if data_io.frame_id_from_filename(file_path, strict=True) in frame_id_set
    ]

    boxes: list[dict[str, Any]] = []
    for file_path in selected_files:
        frame_table = pd.read_csv(file_path, sep=separator)
        boxes.extend(_build_measurement_bounding_boxes(frame_measurements=frame_table))
    return boxes


def _measurement_bbox_metadata(row: Any) -> dict[str, float | int]:
    """Convert one measurement row into the tracking metadata schema."""
    bbox_min = np.asarray([row.bbox_min_x, row.bbox_min_y, row.bbox_min_z], dtype=np.float64)
    bbox_max = np.asarray([row.bbox_max_x, row.bbox_max_y, row.bbox_max_z], dtype=np.float64)
    bbox_extent = np.asarray([row.bbox_dx, row.bbox_dy, row.bbox_dz], dtype=np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)

    if not np.all(np.isfinite(bbox_min)) or not np.all(np.isfinite(bbox_max)) or not np.all(np.isfinite(bbox_extent)):
        return {}
    if np.any(bbox_extent <= 0.0):
        return {}

    return {
        "measurement_centroid_x": float(getattr(row, "centroid_x", np.nan)),
        "measurement_centroid_y": float(getattr(row, "centroid_y", np.nan)),
        "measurement_centroid_z": float(getattr(row, "centroid_z", np.nan)),
        "bbox_mean_center_x": float(bbox_center[0]),
        "bbox_mean_center_y": float(bbox_center[1]),
        "bbox_mean_center_z": float(bbox_center[2]),
        "bbox_dx": float(bbox_extent[0]),
        "bbox_dy": float(bbox_extent[1]),
        "bbox_dz": float(bbox_extent[2]),
        "point_count": int(row.point_count) if hasattr(row, "point_count") else np.nan,
        "bbox_volume": float(row.bbox_volume) if hasattr(row, "bbox_volume") else np.nan,
        "mean_intensity": float(getattr(row, "mean_intensity", np.nan)),
    }


def object_classification(*, current_label: str | None, x: float, y: float, z: float, metadata: dict[str, float | int] | None) -> str:
    """Classify a tracked object from its geometry and current distance."""
    normalized_current_label = _normalize_classification_label(current_label)
    bbox_dx = float((metadata or {}).get("bbox_dx", np.nan))
    bbox_volume = float((metadata or {}).get("bbox_volume", np.nan))
    distance = float(np.sqrt(x * x + y * y + z * z))

    if np.isfinite(bbox_dx) and bbox_dx < TRACKING_HYPERPARAMETERS.pedestrian_bbox_dx_max:
        candidate_label = CLASSIFICATION_PEDESTRIAN
    elif np.isfinite(distance) and distance < TRACKING_HYPERPARAMETERS.unclassified_distance_max:
        candidate_label = CLASSIFICATION_NONE
    elif np.isfinite(bbox_volume) and bbox_volume > TRACKING_HYPERPARAMETERS.vehicle_bbox_volume_min:
        candidate_label = CLASSIFICATION_VEHICLE
    else:
        candidate_label = CLASSIFICATION_CYCLIST

    if CLASSIFICATION_RANK[candidate_label] < CLASSIFICATION_RANK[normalized_current_label]:
        return normalized_current_label
    return candidate_label


def _normalize_classification_label(label: str | None) -> str:
    """Map missing or unknown labels to the unclassified fallback."""
    if label in CLASSIFICATION_RANK:
        return str(label)
    return CLASSIFICATION_NONE


def _build_measurement_bounding_boxes(*, frame_measurements: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert measurement rows into visualization bounding boxes."""
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
        center = np.asarray([row.centroid_x, row.centroid_y, row.centroid_z], dtype=np.float64)
        if not np.all(np.isfinite(center)):
            continue

        extent = np.asarray([row.bbox_dx, row.bbox_dy, row.bbox_dz], dtype=np.float64)
        if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
            continue

        boxes.append(
            {
                "frame_id": int(row.frame_id),
                "center": (float(center[0]), float(center[1]), float(center[2])),
                "extent": (float(extent[0]), float(extent[1]), float(extent[2])),
                "color": MEASUREMENT_BOX_COLOR,
            }
        )
    return boxes


def _build_track_state_bounding_boxes(*, track_table: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert confirmed track states into visualization bounding boxes."""
    required_columns = ("frame_id", "track_id", *TRACK_STATE_COLUMNS)
    missing_columns = [column_name for column_name in required_columns if column_name not in track_table.columns]
    if missing_columns:
        raise ValueError(f"Track table is missing required columns for state track boxes: {missing_columns}")

    boxes: list[dict[str, Any]] = []
    for row in track_table.itertuples(index=False):
        center = np.asarray([row.x, row.y, row.z], dtype=np.float64)
        if not np.all(np.isfinite(center)):
            continue

        extent = np.asarray([row.bbox_dx, row.bbox_dy, row.bbox_dz], dtype=np.float64)
        if not np.all(np.isfinite(extent)) or np.any(extent <= 0.0):
            continue

        track_id = int(row.track_id)
        classification_label = _normalize_classification_label(getattr(row, TRACK_CLASSIFICATION_COLUMN, None))
        boxes.append(
            {
                "frame_id": int(row.frame_id),
                "name": f"track_{track_id}_{classification_label}",
                "center": (float(center[0]), float(center[1]), float(center[2])),
                "extent": (float(extent[0]), float(extent[1]), float(extent[2])),
                "color": STATE_BOX_COLOR,
            }
        )
    return boxes
