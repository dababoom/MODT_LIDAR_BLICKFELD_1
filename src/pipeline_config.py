from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


TABLE_HYPERPARAMETER_COUNT = 38


@dataclass(frozen=True, slots=True)
class PreprocessingHyperparameters:
    sor_nb_neighbors: int = 10
    sor_std_ratio: float = 3.5
    ror_nb_points: int = 3
    ror_search_radius: float = 0.8
    roi_x_min: float = -15.0
    roi_y_min: float = 0.0
    roi_z_min: float = -5.0
    roi_x_max: float = 15.0
    roi_y_max: float = 45.0
    roi_z_max: float = 25.0
    ground_distance_threshold: float = 0.24
    ground_max_sigma: float = 0.25
    ground_num_iterations: int = 20000
    ground_refinement_steps: int = 5
    ground_sample_size: int = 3


@dataclass(frozen=True, slots=True)
class DetectionHyperparameters:
    voxel_size: float = 0.25
    window_size: int = 10
    moving_ratio_max: float = 0.2
    static_ratio_min: float = 0.7
    dbscan_eps: float = 0.45
    dbscan_min_points: int = 5
    min_cluster_points: int = 12


@dataclass(frozen=True, slots=True)
class TrackingHyperparameters:
    max_tracks: int = 2
    birth_distance_m: float = 0.2
    miss_probability_threshold: float = 0.5
    min_hits_confirm: int = 4
    max_missed: int = 3
    dt_seconds: float = 0.41067
    motion_sigma: float = 1.5
    measurement_sigma: float = 0.3
    init_pos_sigma: float = 1.0
    init_vel_sigma: float = 2.5
    prob_detect: float = 0.9
    prob_gate: float = 0.85
    clutter_density: float = 1e-4
    pedestrian_bbox_dx_max: float = 1.13
    unclassified_distance_max: float = 10.0
    vehicle_bbox_volume_min: float = 3.2


PREPROCESSING_HYPERPARAMETERS = PreprocessingHyperparameters()
DETECTION_HYPERPARAMETERS = DetectionHyperparameters()
TRACKING_HYPERPARAMETERS = TrackingHyperparameters()

ROI_MIN_BOUND = np.array(
    [
        PREPROCESSING_HYPERPARAMETERS.roi_x_min,
        PREPROCESSING_HYPERPARAMETERS.roi_y_min,
        PREPROCESSING_HYPERPARAMETERS.roi_z_min,
    ],
    dtype=np.float32,
)
ROI_MAX_BOUND = np.array(
    [
        PREPROCESSING_HYPERPARAMETERS.roi_x_max,
        PREPROCESSING_HYPERPARAMETERS.roi_y_max,
        PREPROCESSING_HYPERPARAMETERS.roi_z_max,
    ],
    dtype=np.float32,
)
MAX_MISSED = TRACKING_HYPERPARAMETERS.max_missed
MIN_HITS_CONFIRM = TRACKING_HYPERPARAMETERS.min_hits_confirm


def count_table_hyperparameters() -> int:
    return (
        len(fields(PreprocessingHyperparameters))
        + len(fields(DetectionHyperparameters))
        + len(fields(TrackingHyperparameters))
    )
