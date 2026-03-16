from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

import data_io
from pipeline_config import PREPROCESSING_HYPERPARAMETERS, PreprocessingHyperparameters
from point_cloud_visualization import visualize_lidar_frames_o3d


PREPROCESSED_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "preprocessed"


def _bool_tensor(mask: np.ndarray) -> o3d.core.Tensor:
    """Convert a NumPy boolean mask into an Open3D boolean tensor."""
    return o3d.core.Tensor(np.asarray(mask, dtype=bool), dtype=o3d.core.Dtype.Bool)


def _iter_frame_ranges(frame_ids: np.ndarray) -> Iterator[tuple[int, int, int]]:
    """Yield contiguous frame ranges as `(frame_id, start, end)` tuples."""
    if frame_ids.size == 0:
        return

    boundaries = np.flatnonzero(frame_ids[1:] != frame_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [frame_ids.size]))
    for start, end in zip(starts, ends):
        yield int(frame_ids[start]), int(start), int(end)


def _slice_point_cloud(point_cloud: o3d.t.geometry.PointCloud, *, start: int, end: int) -> o3d.t.geometry.PointCloud:
    """Create a point-cloud view for one contiguous frame slice."""
    frame_cloud = o3d.t.geometry.PointCloud(point_cloud.point.positions[start:end].clone())
    for attribute_name in list(point_cloud.point):
        if attribute_name == "positions":
            continue
        frame_cloud.point[attribute_name] = point_cloud.point[attribute_name][start:end].clone()
    return frame_cloud


def _select_point_cloud_by_mask(
    point_cloud: o3d.t.geometry.PointCloud,
    mask: np.ndarray,
) -> o3d.t.geometry.PointCloud:
    """Return the masked subset while preserving all point attributes."""
    return point_cloud.select_by_mask(_bool_tensor(mask))


def _outlier_indices_by_frame(frame_ids: np.ndarray, inlier_mask: np.ndarray) -> dict[int, list[int]]:
    """Group local outlier indices by frame id."""
    outlier_indices: dict[int, list[int]] = {}
    for frame_id, start, end in _iter_frame_ranges(frame_ids):
        local_outliers = np.flatnonzero(~inlier_mask[start:end]).tolist()
        if local_outliers:
            outlier_indices[frame_id] = local_outliers
    return outlier_indices


def _apply_framewise_outlier_filter(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    filter_fn: Callable[[o3d.t.geometry.PointCloud], tuple[o3d.t.geometry.PointCloud, o3d.core.Tensor]],
) -> tuple[o3d.t.geometry.PointCloud, o3d.core.Tensor]:
    """Apply an Open3D outlier filter independently to each frame."""
    if "frame_id" not in point_cloud.point:
        return filter_fn(point_cloud)

    point_count = int(point_cloud.point.positions.shape[0])
    if point_count == 0:
        empty_mask = np.zeros((0,), dtype=bool)
        return point_cloud, _bool_tensor(empty_mask)

    frame_id_values = point_cloud.point.frame_id.numpy().reshape(-1).astype(np.int64)
    inlier_mask = np.zeros(point_count, dtype=bool)
    for _, start, end in _iter_frame_ranges(frame_id_values):
        frame_cloud = _slice_point_cloud(point_cloud, start=start, end=end)
        _, frame_inlier_mask = filter_fn(frame_cloud)
        inlier_mask[start:end] = frame_inlier_mask.numpy().reshape(-1).astype(bool)

    return _select_point_cloud_by_mask(point_cloud, inlier_mask), _bool_tensor(inlier_mask)


def data_preprocessing(
    path: str | Path,
    max_files: int | None = None,
    visualization: bool = True,
    hyperparameters: PreprocessingHyperparameters = PREPROCESSING_HYPERPARAMETERS,
) -> o3d.t.geometry.PointCloud:
    """Run the preprocessing pipeline on raw LiDAR CSV files."""
    cfg = hyperparameters
    frames = data_io.load_lidar_data(path, max_files=max_files)
    point_cloud = data_io.convert_pd_to_point_cloud(frames)
    filtered_point_cloud, stat_inlier_mask = _remove_statistical_outliers_by_frame(
        point_cloud,
        nb_neighbors=cfg.sor_nb_neighbors,
        std_ratio=cfg.sor_std_ratio,
    )
    report_point_reduction(point_cloud, filtered_point_cloud, operation_name="remove_statistical_outliers")
    point_cloud_before_radius = filtered_point_cloud
    filtered_point_cloud, radius_inlier_mask = _remove_radius_outliers_by_frame(
        filtered_point_cloud,
        nb_points=cfg.ror_nb_points,
        search_radius=cfg.ror_search_radius,
    )
    report_point_reduction(point_cloud_before_radius, filtered_point_cloud, operation_name="remove_radius_outliers")
    point_cloud_before_crop = filtered_point_cloud
    roi_min_bound = np.array([cfg.roi_x_min, cfg.roi_y_min, cfg.roi_z_min], dtype=np.float32)
    roi_max_bound = np.array([cfg.roi_x_max, cfg.roi_y_max, cfg.roi_z_max], dtype=np.float32)
    filtered_point_cloud, crop_inlier_mask, crop_outlier_indices_on_radius_frames = _crop_point_cloud_by_bounds(
        filtered_point_cloud,
        min_bound=roi_min_bound,
        max_bound=roi_max_bound,
    )
    report_point_reduction(point_cloud_before_crop, filtered_point_cloud, operation_name="crop_point_cloud")
    point_cloud_before_ground = filtered_point_cloud
    ground_random_seed = 42
    ground_print_plane_model = True
    filtered_point_cloud, ground_indices_on_crop_frames = _remove_ground_plane_by_frame_magsac_plus_plus(
        filtered_point_cloud,
        distance_threshold=cfg.ground_distance_threshold,
        max_sigma=cfg.ground_max_sigma,
        num_iterations=cfg.ground_num_iterations,
        refinement_steps=cfg.ground_refinement_steps,
        sample_size=cfg.ground_sample_size,
        random_seed=ground_random_seed,
        print_plane_model=ground_print_plane_model,
    )
    report_point_reduction(point_cloud_before_ground, filtered_point_cloud, operation_name="remove_ground_plane")
    written_preprocessed_files = data_io.save_point_cloud_frames_to_csv(
        filtered_point_cloud,
        output_dir=PREPROCESSED_OUTPUT_DIR,
    )
    print(f"[save_preprocessed_csv] Saved {len(written_preprocessed_files)} files to: {PREPROCESSED_OUTPUT_DIR}")

    if visualization:
        stat_visualization_frames, stat_outlier_indices_by_frame = _build_reduction_visualization_inputs(point_cloud, stat_inlier_mask)
        radius_visualization_frames, radius_outlier_indices_on_stat_frames = _build_reduction_visualization_inputs(point_cloud_before_radius, radius_inlier_mask)
        crop_visualization_frames = data_io.point_cloud_to_dataframe(point_cloud_before_crop)
        radius_outlier_indices_on_original_frames = _remap_subset_indices_to_original_frame_indices(original_frames=stat_visualization_frames, original_inlier_mask=stat_inlier_mask, subset_outlier_indices_by_frame=radius_outlier_indices_on_stat_frames)
        crop_outlier_indices_on_stat_frames = _remap_subset_indices_to_original_frame_indices(original_frames=radius_visualization_frames, original_inlier_mask=radius_inlier_mask, subset_outlier_indices_by_frame=crop_outlier_indices_on_radius_frames)
        crop_outlier_indices_on_original_frames = _remap_subset_indices_to_original_frame_indices(original_frames=stat_visualization_frames, original_inlier_mask=stat_inlier_mask, subset_outlier_indices_by_frame=crop_outlier_indices_on_stat_frames)
        ground_outlier_indices_on_radius_frames = _remap_subset_indices_to_original_frame_indices(original_frames=crop_visualization_frames, original_inlier_mask=crop_inlier_mask, subset_outlier_indices_by_frame=ground_indices_on_crop_frames)
        ground_outlier_indices_on_stat_frames = _remap_subset_indices_to_original_frame_indices(original_frames=radius_visualization_frames, original_inlier_mask=radius_inlier_mask, subset_outlier_indices_by_frame=ground_outlier_indices_on_radius_frames)
        ground_outlier_indices_on_original_frames = _remap_subset_indices_to_original_frame_indices(original_frames=stat_visualization_frames, original_inlier_mask=stat_inlier_mask, subset_outlier_indices_by_frame=ground_outlier_indices_on_stat_frames)

        visualize_lidar_frames_o3d(
            stat_visualization_frames,
            color_mode="single",
            point_color=(0.6, 0.6, 0.6),
            flagged_indices_by_frame=[stat_outlier_indices_by_frame, radius_outlier_indices_on_original_frames, crop_outlier_indices_on_original_frames, ground_outlier_indices_on_original_frames],
            flagged_color=[(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.0, 0.6, 1.0), (0.0, 1.0, 0.0)],
            max_frames=10000,
            camera_config_name="wide_eagle",
            window_title="after statistical, radius, crop and ground filtering"
        )

    return filtered_point_cloud


def _build_reduction_visualization_inputs(
    point_cloud: o3d.t.geometry.PointCloud, inlier_mask: o3d.core.Tensor
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    """Create visualization tables and local outlier indices for one reduction step."""
    visualization_frames = data_io.point_cloud_to_dataframe(point_cloud)
    inliers = inlier_mask.numpy().reshape(-1).astype(bool)
    point_count = int(len(visualization_frames))
    if inliers.size != point_count:
        raise ValueError("Inlier mask length does not match point cloud point count.")

    frame_ids = visualization_frames["frame_id"].to_numpy(dtype=np.int64, copy=False)
    return visualization_frames, _outlier_indices_by_frame(frame_ids, inliers)


def _remove_statistical_outliers_by_frame(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    nb_neighbors: int,
    std_ratio: float,
) -> tuple[o3d.t.geometry.PointCloud, o3d.core.Tensor]:
    """Remove statistical outliers independently in each frame."""
    return _apply_framewise_outlier_filter(
        point_cloud,
        filter_fn=lambda frame_cloud: frame_cloud.remove_statistical_outliers(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        ),
    )


def _remove_radius_outliers_by_frame(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    nb_points: int,
    search_radius: float,
) -> tuple[o3d.t.geometry.PointCloud, o3d.core.Tensor]:
    """Remove radius outliers independently in each frame."""
    return _apply_framewise_outlier_filter(
        point_cloud,
        filter_fn=lambda frame_cloud: frame_cloud.remove_radius_outliers(
            nb_points=nb_points,
            search_radius=search_radius,
        ),
    )


def _crop_point_cloud_by_bounds(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> tuple[o3d.t.geometry.PointCloud, o3d.core.Tensor, dict[int, list[int]]]:
    """Keep only points inside the ROI bounds and report removed indices per frame."""
    min_bound_array = np.asarray(min_bound, dtype=np.float32).reshape(-1)
    max_bound_array = np.asarray(max_bound, dtype=np.float32).reshape(-1)

    positions = point_cloud.point.positions.numpy()
    inlier_mask = np.logical_and(positions >= min_bound_array, positions <= max_bound_array).all(axis=1)
    frame_ids = point_cloud.point.frame_id.numpy().reshape(-1).astype(np.int64)
    return (
        _select_point_cloud_by_mask(point_cloud, inlier_mask),
        _bool_tensor(inlier_mask),
        _outlier_indices_by_frame(frame_ids, inlier_mask),
    )


def _remove_ground_plane_by_frame(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    target_confidence_p: float,
    random_seed: int,
    print_plane_model: bool = False,
) -> tuple[o3d.t.geometry.PointCloud, dict[int, list[int]]]:
    """Remove one RANSAC ground plane from each frame."""
    base_seed = int(random_seed)
    positions = point_cloud.point.positions.numpy()
    point_count = int(positions.shape[0])
    frame_id_values = point_cloud.point.frame_id.numpy().reshape(-1).astype(np.int64)

    keep_mask = np.ones(point_count, dtype=bool)
    ground_indices_by_frame: dict[int, list[int]] = {}
    for frame_id, start, end in _iter_frame_ranges(frame_id_values):
        frame_seed = (base_seed + frame_id) % (2**31 - 1)
        o3d.utility.random.seed(int(frame_seed))
        frame_positions = positions[start:end]
        frame_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(frame_positions, dtype=o3d.core.Dtype.Float32))
        frame_plane_model, frame_ground_indices = frame_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations, probability=target_confidence_p)
        frame_ground_indices_array = frame_ground_indices.numpy().reshape(-1).astype(np.int64)
        if frame_ground_indices_array.size == 0:
            continue
        if print_plane_model:
            plane_model = frame_plane_model.numpy().reshape(-1).astype(np.float64)
            frame_ground_points = frame_positions[frame_ground_indices_array]
            print(
                f"[ground_plane_model][frame {frame_id}] "
                f"a={plane_model[0]:.9f}, b={plane_model[1]:.9f}, c={plane_model[2]:.9f}, d={plane_model[3]:.9f}"
            )
            representative_indices = np.array(
                [0, frame_ground_points.shape[0] // 2, frame_ground_points.shape[0] - 1], dtype=np.int64
            )
            representative_points = frame_ground_points[representative_indices]
            with np.printoptions(suppress=True, linewidth=200):
                print(
                    f"[ground_plane_representative_points][frame {frame_id}] "
                    "(three inlier coordinates on the fitted plane)"
                )
                print(representative_points)
        keep_mask[start + frame_ground_indices_array] = False
        ground_indices_by_frame[frame_id] = frame_ground_indices_array.tolist()

    return _select_point_cloud_by_mask(point_cloud, keep_mask), ground_indices_by_frame


def _remove_ground_plane_by_frame_magsac_plus_plus(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    distance_threshold: float,
    max_sigma: float,
    num_iterations: int,
    refinement_steps: int = 3,
    sample_size: int = 3,
    random_seed: int | None = None,
    print_plane_model: bool = False,
) -> tuple[o3d.t.geometry.PointCloud, dict[int, list[int]]]:
    """Remove per-frame ground using a MAGSAC++-inspired robust plane estimator.

    Method summary:
    1. For each frame, sample a small point set repeatedly to form candidate planes.
    2. Score candidates by soft sigma-consensus weights computed from
       point-to-plane distances normalized by `max_sigma`.
    3. Select the plane with the largest summed weight and refine it via
       iterative weighted SVD plane fitting.
    4. Classify final inliers using a hard `distance_threshold` and remove
       those points as ground.

    Note:
    This is inspired by MAGSAC++ but does not implement the original
    marginalized-sigma likelihood scoring exactly.
    """
    if distance_threshold <= 0:
        raise ValueError("distance_threshold must be > 0.")
    if max_sigma <= 0:
        raise ValueError("max_sigma must be > 0.")
    if num_iterations <= 0:
        raise ValueError("num_iterations must be > 0.")
    if refinement_steps <= 0:
        raise ValueError("refinement_steps must be > 0.")
    if sample_size < 3:
        raise ValueError("sample_size must be >= 3.")

    positions = point_cloud.point.positions.numpy().astype(np.float64, copy=False)
    point_count = int(positions.shape[0])
    if point_count == 0:
        return point_cloud, {}
    frame_id_values = point_cloud.point.frame_id.numpy().reshape(-1).astype(np.int64)

    keep_mask = np.ones(point_count, dtype=bool)
    ground_indices_by_frame: dict[int, list[int]] = {}
    for frame_id, start, end in _iter_frame_ranges(frame_id_values):
        frame_positions = positions[start:end]
        if frame_positions.shape[0] < sample_size:
            continue

        frame_seed = None
        if random_seed is not None:
            frame_seed = (int(random_seed) + frame_id) % (2**31 - 1)
        frame_rng = np.random.default_rng(seed=frame_seed)

        frame_plane_model, frame_ground_indices_array = _fit_plane_magsacpp(
            frame_positions,
            distance_threshold=distance_threshold,
            max_sigma=max_sigma,
            num_iterations=num_iterations,
            refinement_steps=refinement_steps,
            sample_size=sample_size,
            rng=frame_rng,
        )
        if frame_plane_model is None or frame_ground_indices_array.size == 0:
            continue

        if print_plane_model:
            print(
                f"[ground_plane_model_magsacpp][frame {frame_id}] "
                f"a={frame_plane_model[0]:.9f}, b={frame_plane_model[1]:.9f}, "
                f"c={frame_plane_model[2]:.9f}, d={frame_plane_model[3]:.9f}"
            )

        keep_mask[start + frame_ground_indices_array] = False
        ground_indices_by_frame[frame_id] = frame_ground_indices_array.tolist()

    return _select_point_cloud_by_mask(point_cloud, keep_mask), ground_indices_by_frame


def _fit_plane_magsacpp(
    points_xyz: np.ndarray,
    *,
    distance_threshold: float,
    max_sigma: float,
    num_iterations: int,
    refinement_steps: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Fit one frame plane with random minimal sampling and soft consensus.

    The best candidate is selected by maximizing S = sum_i w_i, where w_i
    comes from sigma-normalized point-to-plane distances. The selected plane is
    then refined via iterative weighted SVD and finally thresholded to obtain
    binary ground inliers.
    """
    point_count = int(points_xyz.shape[0])
    if point_count < sample_size:
        return None, np.empty((0,), dtype=np.int64)

    best_plane: np.ndarray | None = None
    best_score = -np.inf

    for _ in range(num_iterations):
        # Minimal sample for plane hypothesis generation.
        sample_indices = rng.choice(point_count, size=sample_size, replace=False)
        candidate_plane = _plane_from_sampled_points(points_xyz[sample_indices])
        if candidate_plane is None:
            continue
        # Point-to-plane distances for all points under this candidate.
        residuals = _point_to_plane_distances(points_xyz, candidate_plane)
        # Soft inlier scoring (sum of sigma-consensus weights).
        score, _ = _magsacpp_sigma_consensus(residuals, max_sigma=max_sigma)
        if score > best_score:
            best_score = score
            best_plane = candidate_plane

    if best_plane is None:
        return None, np.empty((0,), dtype=np.int64)

    refined_plane = _refine_plane_weighted(
        points_xyz,
        best_plane,
        max_sigma=max_sigma,
        refinement_steps=refinement_steps,
    )
    refined_residuals = _point_to_plane_distances(points_xyz, refined_plane)
    inlier_indices = np.flatnonzero(refined_residuals <= distance_threshold).astype(np.int64)
    return refined_plane, inlier_indices


def _plane_from_sampled_points(points_xyz: np.ndarray) -> np.ndarray | None:
    """Estimate a plane model from sampled 3D points."""
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3).")
    if points_xyz.shape[0] < 3:
        raise ValueError("points_xyz must contain at least 3 points.")

    if points_xyz.shape[0] == 3:
        p0, p1, p2 = points_xyz
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return None
        normal = normal / normal_norm
        point_on_plane = p0
    else:
        centroid = points_xyz.mean(axis=0)
        centered_points = points_xyz - centroid
        _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
        normal = vh[-1, :]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return None
        normal = normal / normal_norm
        point_on_plane = centroid

    # Keep orientation stable across runs.
    if normal[2] < 0:
        normal = -normal
    d = -float(np.dot(normal, point_on_plane))
    return np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)


def _point_to_plane_distances(points_xyz: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """Compute absolute point-to-plane distances for one plane hypothesis."""
    normal = plane_model[:3]
    d = float(plane_model[3])
    return np.abs(points_xyz @ normal + d)


def _magsacpp_sigma_consensus(residuals: np.ndarray, *, max_sigma: float) -> tuple[float, np.ndarray]:
    """Return sigma-consensus style score and per-point soft weights.

    Distances are normalized as s_i = d_i / max_sigma and mapped with a
    Tukey-type biweight kernel:
    w_i = (1 - s_i^2)^2 for s_i < 1, else 0.
    """
    scaled = residuals / max_sigma
    weights = np.where(scaled < 1.0, (1.0 - scaled**2) ** 2, 0.0)
    return float(np.sum(weights)), weights


def _refine_plane_weighted(
    points_xyz: np.ndarray,
    plane_model: np.ndarray,
    *,
    max_sigma: float,
    refinement_steps: int,
) -> np.ndarray:
    """Refine a plane by iterative weighted SVD (weighted PCA).

    Per iteration:
    1. Recompute point weights from current point-to-plane distances.
    2. Compute weighted centroid mu = sum(w_i * p_i) / sum(w_i).
    3. Build weighted centered matrix X with rows
       q_i = sqrt(w_i) * (p_i - mu), shape (N, 3).
    4. Run SVD(X) = U * Sigma * V^T and use the right singular vector
       associated with the smallest singular value as the plane normal.
    """
    refined_plane = plane_model.astype(np.float64, copy=True)
    for _ in range(max(1, int(refinement_steps))):
        residuals = _point_to_plane_distances(points_xyz, refined_plane)
        _, weights = _magsacpp_sigma_consensus(residuals, max_sigma=max_sigma)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            break

        # Weighted centroid of the 3D point cloud (a single point in R^3).
        centroid = np.sum(points_xyz * weights[:, None], axis=0) / weight_sum
        centered_points = points_xyz - centroid
        # Weighted centered points: rows q_i = sqrt(w_i) * (p_i - mu).
        weighted_points = centered_points * np.sqrt(weights)[:, None]
        # SVD gives orthogonal directions; smallest-spread direction is the normal.
        _, _, vh = np.linalg.svd(weighted_points, full_matrices=False)
        normal = vh[-1, :]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            break

        normal = normal / normal_norm
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, centroid))
        refined_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)

    return refined_plane


def _remove_ground_plane_by_frame_planar_patches(
    point_cloud: o3d.t.geometry.PointCloud,
    *,
    normal_variance_threshold_deg: float = 60.0, # maximum variance in degrees of the normals of points in a planar patch for it to be accepted as a ground plane candidate
    coplanarity_deg: float = 57.0, # maximum angle in degrees between the normals of points in a planar patch for them to be considered coplanar
    outlier_ratio: float = 0.87, # maximum ratio of points in a planar patch that can be outliers (i.e., not within the normal variance threshold) for the patch to be accepted as a ground plane candidate
    min_plane_edge_length: float = 1.0, # minimum length of the edges of the planar patch bounding box
    min_num_points: int = 1000, # minimum number of points in a planar patch for it to be accepted as a ground plane candidate
    knn: int = 35, # number of nearest neighbors to use for normal estimation and planar patch detection
) -> tuple[o3d.t.geometry.PointCloud, dict[int, list[int]]]:
    """Alternative ground-removal path based on planar patch detection."""
    positions = point_cloud.point.positions.numpy()
    point_count = int(positions.shape[0])
    if point_count == 0:
        return point_cloud, {}
    frame_id_values = point_cloud.point.frame_id.numpy().reshape(-1).astype(np.int64)

    keep_mask = np.ones(point_count, dtype=bool)
    ground_indices_by_frame: dict[int, list[int]] = {}
    for frame_id, start, end in _iter_frame_ranges(frame_id_values):
        frame_positions = positions[start:end]
        if frame_positions.shape[0] < 3:
            continue

        legacy_frame_cloud = o3d.geometry.PointCloud()
        legacy_frame_cloud.points = o3d.utility.Vector3dVector(frame_positions.astype(np.float64))
        legacy_frame_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        planar_patches = legacy_frame_cloud.detect_planar_patches(
            normal_variance_threshold_deg=normal_variance_threshold_deg,
            coplanarity_deg=coplanarity_deg,
            outlier_ratio=outlier_ratio,
            min_plane_edge_length=min_plane_edge_length,
            min_num_points=min_num_points,
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn),
        )
        if not planar_patches:
            continue

        largest_patch_indices: np.ndarray | None = None
        for patch in planar_patches:
            patch_indices = np.asarray(
                patch.get_point_indices_within_bounding_box(legacy_frame_cloud.points), dtype=np.int64
            ).reshape(-1)
            if patch_indices.size == 0:
                continue
            if largest_patch_indices is None or patch_indices.size > largest_patch_indices.size:
                largest_patch_indices = patch_indices

        if largest_patch_indices is None or largest_patch_indices.size == 0:
            continue

        keep_mask[start + largest_patch_indices] = False
        ground_indices_by_frame[frame_id] = largest_patch_indices.tolist()

    return _select_point_cloud_by_mask(point_cloud, keep_mask), ground_indices_by_frame


def _remap_subset_indices_to_original_frame_indices(
    *,
    original_frames: pd.DataFrame,
    original_inlier_mask: o3d.core.Tensor,
    subset_outlier_indices_by_frame: dict[int, list[int]],
) -> dict[int, list[int]]:
    """Map outlier indices from a filtered subset back to the original frame-local indices."""
    inliers = original_inlier_mask.numpy().reshape(-1).astype(bool)
    if inliers.size != len(original_frames):
        raise ValueError("Inlier mask length does not match original frame table.")

    frame_ids = original_frames["frame_id"].to_numpy(dtype=np.int64, copy=False)
    remapped_outliers: dict[int, list[int]] = {}

    for frame_id, subset_indices in subset_outlier_indices_by_frame.items():
        frame_mask = frame_ids == frame_id
        original_inlier_local_indices = np.flatnonzero(inliers[frame_mask])
        mapped_indices = []
        for subset_index in subset_indices:
            int_index = int(subset_index)
            if 0 <= int_index < original_inlier_local_indices.size:
                mapped_indices.append(int(original_inlier_local_indices[int_index]))
        if mapped_indices:
            remapped_outliers[int(frame_id)] = sorted(set(mapped_indices))

    return remapped_outliers


def report_point_reduction(before_point_cloud: o3d.t.geometry.PointCloud, after_point_cloud: o3d.t.geometry.PointCloud, operation_name: str) -> None:
    """Print the number and percentage of removed points for one preprocessing step."""
    total_points = int(before_point_cloud.point.positions.shape[0])
    remaining_points = int(after_point_cloud.point.positions.shape[0])
    removed_points = total_points - remaining_points
    removal_rate = (removed_points / total_points) if total_points else 0.0
    removed_percent = removal_rate * 100.0
    print(
        f"[{operation_name}] Outlier rate: {removed_points}/{total_points} ({removal_rate:.2%}). "
        f"Deleted points: {removed_percent:.2f}%"
    )
