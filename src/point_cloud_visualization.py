from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd

CAMERA_VIEW_CONFIGS: dict[str, dict[str, tuple[float, float, float] | float]] = {
    "close_sensor": {
        "lookat": (0.0, 0.0, 0.0),
        "eye": (0.0, -3.0, -1.5),
        "up": (0.0, 0.0, 0.6),
        "rot_x_deg": -45.0,
    },
    "wide_eagle": {
        "lookat": (0.0, 20.0, 0.0),
        "eye": (0.0, -6.0, -1.0),
        "up": (0.0, 0.0, 0.8),
        "rot_x_deg": -40.0,
    },
}


def visualize_lidar_frames_o3d(
    frames: pd.DataFrame,
    *,
    frame_column: str = "frame_id",
    x_column: str | None = None,
    y_column: str | None = None,
    z_column: str | None = None,
    intensity_column: str | None = None,
    frame_ids: Sequence[int] | None = None,
    max_frames: int | None = 10,
    frame_stride: int = 1,
    color_mode: str = "intensity",
    point_color: Sequence[float] = (0.65, 0.65, 0.65),
    point_size: float = 2.0,
    flagged_indices_by_frame: Mapping[int, Sequence[int]] | Sequence[Mapping[int, Sequence[int]]] | None = None,
    flagged_points_xyz: np.ndarray | Sequence[Sequence[float]] | None = None,
    flagged_color: Sequence[float] | Sequence[Sequence[float]] = (1.0, 0.2, 0.1),
    flagged_point_size: float = 2.0,
    exclude_flagged_from_frames: bool = True,
    bounding_boxes: Sequence[Mapping[str, Any]] | None = None,
    show_box_names: bool = True,
    sync_labels_with_frames: bool = False,
    default_box_color: Sequence[float] = (0.1, 0.9, 0.1),
    box_line_width: float = 2.0,
    show_frame_labels: bool = False,
    frame_label_height_offset: float = 0.5,
    frame_time_step: int = 0.2,
    auto_play: bool = False,
    window_title: str = "LiDAR Frames - O3DVisualizer",
    width: int = 1400,
    height: int = 900,
    background_rgba: Sequence[float] = (0.05, 0.05, 0.05, 1.0),
    camera_config_name: str = "wide_eagle",
    lookat: Sequence[float] | None = None,
    eye: Sequence[float] | None = None,
    up: Sequence[float] | None = None,
    show_settings: bool = True,
    show_ground: bool = False,
    show_skybox: bool = False,
) -> None:
    """Visualize LiDAR frames with Open3D's O3DVisualizer.

    The input DataFrame is expected to contain XYZ coordinates and can optionally contain
    intensity and frame id columns. Useful optional structures:
    - flagged_indices_by_frame: {frame_id: [local_point_index, ...]} or
      [{frame_id: [local_point_index, ...]}, ...]
    - flagged_color: one RGB triplet for all flagged sets, or one RGB triplet per flagged set
    - exclude_flagged_from_frames: if True, flagged indices are removed from the "frames" group
    - flagged_points_xyz: Nx3 coordinates that should be highlighted
    - bounding_boxes: list of dicts with:
      center (or cx/cy/cz), extent (or dx/dy/dz), optional yaw/rotation_matrix/R,
      optional frame_id, optional color, optional name
    - sync_labels_with_frames: if True, 3D labels (frame labels / box names) are shown
      only for the currently visible frame.
    """
    if frames.empty:
        raise ValueError("frames is empty; nothing to visualize.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be > 0.")
    if frame_time_step <= 0:
        raise ValueError("frame_time_step must be > 0.")
    if point_size <= 0:
        raise ValueError("point_size must be > 0.")
    if flagged_point_size <= 0:
        raise ValueError("flagged_point_size must be > 0.")
    if box_line_width <= 0:
        raise ValueError("box_line_width must be > 0.")
    if camera_config_name not in CAMERA_VIEW_CONFIGS:
        raise ValueError(
            f"Unknown camera_config_name: {camera_config_name}. Available: {sorted(CAMERA_VIEW_CONFIGS)}"
        )
    camera_config = CAMERA_VIEW_CONFIGS[camera_config_name]
    camera_lookat = np.asarray(camera_config["lookat"] if lookat is None else lookat, dtype=np.float32)
    camera_eye = np.asarray(camera_config["eye"] if eye is None else eye, dtype=np.float32)
    camera_up = np.asarray(camera_config["up"] if up is None else up, dtype=np.float32)
    if camera_lookat.shape != (3,) or camera_eye.shape != (3,) or camera_up.shape != (3,):
        raise ValueError("lookat, eye, and up must be length-3 vectors.")
    rot_x_deg = float(camera_config.get("rot_x_deg", 0.0))
    rot_x_rad = np.deg2rad(rot_x_deg)
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
            [0.0, np.sin(rot_x_rad), np.cos(rot_x_rad)],
        ],
        dtype=np.float32,
    )
    camera_eye = camera_lookat + rot_x @ (camera_eye - camera_lookat)
    camera_up = rot_x @ camera_up

    # O3DVisualizer GUI properties are integer-typed in Open3D 0.19.
    gui_point_size = max(1, int(round(point_size)))
    gui_line_width = max(1, int(round(box_line_width)))

    x_col = _resolve_column_name(frames, x_column, "x")
    y_col = _resolve_column_name(frames, y_column, "y")
    z_col = _resolve_column_name(frames, z_column, "z")
    intensity_col = _resolve_column_name(frames, intensity_column, "intensity", required=False)
    has_frame_column = _has_column_ci(frames, frame_column)
    frame_col = _resolve_column_name(frames, frame_column, frame_column, required=False) if has_frame_column else None

    numeric = frames.copy()
    numeric_columns = [x_col, y_col, z_col] + ([intensity_col] if intensity_col is not None else [])
    for column in numeric_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric = numeric.dropna(subset=numeric_columns)

    if frame_col is None:
        numeric["_generated_frame_id"] = 0
        frame_col = "_generated_frame_id"
    numeric[frame_col] = pd.to_numeric(numeric[frame_col], errors="coerce")
    numeric = numeric.dropna(subset=[frame_col])
    numeric[frame_col] = numeric[frame_col].astype(int)

    selected_frames = _select_frames(
        available_frame_ids=sorted(numeric[frame_col].unique().tolist()),
        frame_ids=frame_ids,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    if not selected_frames:
        raise ValueError("No frames selected for visualization.")

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    visualizer = o3d.visualization.O3DVisualizer(window_title, width, height)
    visualizer.show_settings = show_settings
    visualizer.show_ground = show_ground
    visualizer.show_skybox(show_skybox)
    visualizer.set_background(np.asarray(background_rgba, dtype=np.float32), None)
    visualizer.point_size = gui_point_size
    visualizer.line_width = gui_line_width
    visualizer.animation_time_step = frame_time_step

    frame_material = o3d.visualization.rendering.MaterialRecord()
    frame_material.shader = "defaultUnlit"
    frame_material.point_size = float(point_size)

    frame_time_lookup = {frame_id: i * frame_time_step for i, frame_id in enumerate(selected_frames)}
    dynamic_labels_by_frame: dict[int, list[tuple[np.ndarray, str]]] = {}
    flagged_layers = _normalize_flagged_layers(flagged_indices_by_frame, flagged_color)
    primary_flagged_color = flagged_layers[0][1] if flagged_layers else _normalize_primary_flagged_color(flagged_color)

    for frame_id in selected_frames:
        frame_data = numeric.loc[numeric[frame_col] == frame_id]
        positions = frame_data[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
        if positions.size == 0:
            continue

        flagged_frame_sets: list[tuple[int, list[int]]] = []
        combined_flagged_indices: set[int] = set()
        for layer_index, (flagged_map, _) in enumerate(flagged_layers):
            if frame_id not in flagged_map:
                continue
            layer_indices = _sanitize_indices(flagged_map[frame_id], len(frame_data))
            if not layer_indices:
                continue
            flagged_frame_sets.append((layer_index, layer_indices))
            combined_flagged_indices.update(layer_indices)

        frame_positions = positions
        frame_data_for_colors = frame_data
        if exclude_flagged_from_frames and combined_flagged_indices:
            keep_mask = np.ones(len(frame_data), dtype=bool)
            keep_mask[list(combined_flagged_indices)] = False
            frame_positions = positions[keep_mask]
            frame_data_for_colors = frame_data.iloc[keep_mask]

        if frame_positions.size > 0:
            point_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(frame_positions, dtype=o3d.core.Dtype.Float32))
            colors = _build_colors(
                frame_data=frame_data_for_colors,
                color_mode=color_mode,
                intensity_column=intensity_col,
                z_column=z_col,
                point_color=point_color,
            )
            point_cloud.point.colors = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)

            geometry_name = f"frame_{frame_id}"
            visualizer.add_geometry(
                name=geometry_name,
                geometry=point_cloud,
                material=frame_material,
                time=frame_time_lookup[frame_id],
                group="frames",
            )

        if show_frame_labels:
            max_z = float(np.max(positions[:, 2]))
            label_pos = np.array([positions[0, 0], positions[0, 1], max_z + frame_label_height_offset], dtype=np.float32)
            if sync_labels_with_frames:
                dynamic_labels_by_frame.setdefault(int(frame_id), []).append((label_pos, f"frame {frame_id}"))
            else:
                visualizer.add_3d_label(label_pos, f"frame {frame_id}")

        for layer_index, flagged_indices in flagged_frame_sets:
            flagged_positions = positions[flagged_indices]
            flagged_cloud = o3d.t.geometry.PointCloud(
                o3d.core.Tensor(flagged_positions, dtype=o3d.core.Dtype.Float32)
            )
            flagged_colors = np.tile(flagged_layers[layer_index][1], (flagged_positions.shape[0], 1)).astype(np.float32)
            flagged_cloud.point.colors = o3d.core.Tensor(flagged_colors, dtype=o3d.core.Dtype.Float32)

            flagged_material = o3d.visualization.rendering.MaterialRecord()
            flagged_material.shader = "defaultUnlit"
            flagged_material.point_size = flagged_point_size

            visualizer.add_geometry(
                name=f"flagged_{layer_index}_frame_{frame_id}",
                geometry=flagged_cloud,
                material=flagged_material,
                time=frame_time_lookup[frame_id],
                group="flags",
            )

    if flagged_points_xyz is not None:
        explicit_flagged = np.asarray(flagged_points_xyz, dtype=np.float32)
        if explicit_flagged.ndim != 2 or explicit_flagged.shape[1] != 3:
            raise ValueError("flagged_points_xyz must have shape (N, 3).")
        if explicit_flagged.shape[0] > 0:
            flagged_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(explicit_flagged, dtype=o3d.core.Dtype.Float32))
            flagged_colors = np.tile(primary_flagged_color, (explicit_flagged.shape[0], 1)).astype(np.float32)
            flagged_cloud.point.colors = o3d.core.Tensor(flagged_colors, dtype=o3d.core.Dtype.Float32)

            flagged_material = o3d.visualization.rendering.MaterialRecord()
            flagged_material.shader = "defaultUnlit"
            flagged_material.point_size = flagged_point_size

            visualizer.add_geometry(
                name="flagged_points_xyz",
                geometry=flagged_cloud,
                material=flagged_material,
                time=0.0,
                group="flags",
            )

    if bounding_boxes:
        box_labels = _add_bounding_boxes_to_visualizer(
            visualizer=visualizer,
            bounding_boxes=bounding_boxes,
            frame_time_lookup=frame_time_lookup,
            default_box_color=default_box_color,
            show_box_names=show_box_names,
            sync_labels_with_frames=sync_labels_with_frames,
            box_line_width=box_line_width,
        )
        if sync_labels_with_frames and box_labels:
            for frame_id, entries in box_labels.items():
                dynamic_labels_by_frame.setdefault(int(frame_id), []).extend(entries)

    if sync_labels_with_frames and dynamic_labels_by_frame:
        _configure_dynamic_labels(
            visualizer=visualizer,
            selected_frames=selected_frames,
            frame_time_step=frame_time_step,
            labels_by_frame=dynamic_labels_by_frame,
        )

    visualizer.setup_camera(60.0, camera_lookat, camera_eye, camera_up)
    visualizer.current_time = 0.0
    visualizer.is_animating = auto_play

    app.add_window(visualizer)
    app.run()


def _resolve_column_name(
    frames: pd.DataFrame,
    explicit_name: str | None,
    canonical_name: str,
    *,
    required: bool = True,
) -> str | None:
    if explicit_name is not None:
        if explicit_name not in frames.columns:
            if required:
                raise ValueError(f"Column not found: {explicit_name}")
            return None
        return explicit_name

    lookup = {column.lower(): column for column in frames.columns}
    match = lookup.get(canonical_name.lower())
    if match is None and required:
        raise ValueError(f"Required column not found: {canonical_name}")
    return match


def _has_column_ci(frames: pd.DataFrame, name: str) -> bool:
    lookup = {column.lower() for column in frames.columns}
    return name.lower() in lookup


def _select_frames(
    available_frame_ids: Sequence[int],
    frame_ids: Sequence[int] | None,
    max_frames: int | None,
    frame_stride: int,
) -> list[int]:
    selected = list(available_frame_ids)
    if frame_ids is not None:
        requested = set(int(frame_id) for frame_id in frame_ids)
        selected = [frame_id for frame_id in selected if frame_id in requested]
    selected = selected[::frame_stride]

    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0 when provided.")
        selected = selected[:max_frames]
    return selected


def _build_colors(
    frame_data: pd.DataFrame,
    color_mode: str,
    intensity_column: str | None,
    z_column: str,
    point_color: Sequence[float],
) -> np.ndarray:
    mode = color_mode.lower()
    if mode == "single":
        return np.tile(_to_rgb(point_color), (len(frame_data), 1)).astype(np.float32)

    if mode == "intensity":
        if intensity_column is None:
            return np.tile(_to_rgb(point_color), (len(frame_data), 1)).astype(np.float32)
        values = frame_data[intensity_column].to_numpy(dtype=np.float32)
        normalized = _robust_normalize(values)
        return np.column_stack((normalized, normalized, normalized)).astype(np.float32)

    if mode == "height":
        values = frame_data[z_column].to_numpy(dtype=np.float32)
        normalized = _robust_normalize(values)
        return np.column_stack((normalized, 0.35 + 0.65 * normalized, 1.0 - normalized)).astype(np.float32)

    raise ValueError("color_mode must be one of: 'intensity', 'height', 'single'.")


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.percentile(values, 2.0))
    high = float(np.percentile(values, 98.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _sanitize_indices(indices: Sequence[int], point_count: int) -> list[int]:
    valid = []
    for idx in indices:
        int_idx = int(idx)
        if 0 <= int_idx < point_count:
            valid.append(int_idx)
    return sorted(set(valid))


def _normalize_flagged_layers(
    flagged_indices_by_frame: Mapping[int, Sequence[int]] | Sequence[Mapping[int, Sequence[int]]] | None,
    flagged_color: Sequence[float] | Sequence[Sequence[float]],
) -> list[tuple[Mapping[int, Sequence[int]], np.ndarray]]:
    if flagged_indices_by_frame is None:
        return []

    if isinstance(flagged_indices_by_frame, Mapping):
        flagged_maps = [flagged_indices_by_frame]
    else:
        flagged_maps = list(flagged_indices_by_frame)
        if not all(isinstance(flagged_map, Mapping) for flagged_map in flagged_maps):
            raise ValueError(
                "flagged_indices_by_frame must be a mapping or a sequence of mappings."
            )

    colors = _normalize_flagged_colors(flagged_color, len(flagged_maps))
    return list(zip(flagged_maps, colors))


def _normalize_flagged_colors(
    flagged_color: Sequence[float] | Sequence[Sequence[float]],
    expected_count: int,
) -> list[np.ndarray]:
    if expected_count <= 0:
        return []

    if _is_single_rgb_color(flagged_color):
        rgb = _to_rgb(flagged_color)  # type: ignore[arg-type]
        return [rgb.copy() for _ in range(expected_count)]

    colors = [_to_rgb(color) for color in flagged_color]  # type: ignore[arg-type]
    if len(colors) != expected_count:
        raise ValueError(
            f"When passing multiple flagged colors, provide exactly one color per flagged set ({expected_count})."
        )
    return colors


def _normalize_primary_flagged_color(flagged_color: Sequence[float] | Sequence[Sequence[float]]) -> np.ndarray:
    if _is_single_rgb_color(flagged_color):
        return _to_rgb(flagged_color)  # type: ignore[arg-type]

    colors = [_to_rgb(color) for color in flagged_color]  # type: ignore[arg-type]
    if not colors:
        raise ValueError("flagged_color must contain at least one RGB value.")
    return colors[0]


def _is_single_rgb_color(flagged_color: Sequence[float] | Sequence[Sequence[float]]) -> bool:
    try:
        rgb = np.asarray(flagged_color, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return rgb.shape == (3,)


def _to_rgb(color: Sequence[float]) -> np.ndarray:
    rgb = np.asarray(color, dtype=np.float32)
    if rgb.shape != (3,):
        raise ValueError("RGB color values must have exactly 3 floats.")
    return np.clip(rgb, 0.0, 1.0)


def _add_bounding_boxes_to_visualizer(
    *,
    visualizer: o3d.visualization.O3DVisualizer,
    bounding_boxes: Sequence[Mapping[str, Any]],
    frame_time_lookup: Mapping[int, float],
    default_box_color: Sequence[float],
    show_box_names: bool,
    sync_labels_with_frames: bool,
    box_line_width: float,
) -> dict[int, list[tuple[np.ndarray, str]]]:
    default_rgb = _to_rgb(default_box_color).tolist()
    labels_by_frame: dict[int, list[tuple[np.ndarray, str]]] = {}
    default_frame_id = int(next(iter(frame_time_lookup.keys())))

    box_material = o3d.visualization.rendering.MaterialRecord()
    box_material.shader = "unlitLine"
    box_material.line_width = box_line_width

    for index, box in enumerate(bounding_boxes):
        center = _extract_xyz(box, preferred_key="center", fallback_keys=("cx", "cy", "cz"))
        extent = _extract_xyz(box, preferred_key="extent", fallback_keys=("dx", "dy", "dz"), positive_only=True)
        rotation = _extract_rotation_matrix(box)

        obb = o3d.geometry.OrientedBoundingBox(center, rotation, extent)
        line_set_legacy = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)

        rgb = _to_rgb(box.get("color", default_rgb)).tolist()
        line_set_legacy.paint_uniform_color(rgb)
        line_set = o3d.t.geometry.LineSet.from_legacy(line_set_legacy)

        frame_id = box.get("frame_id")
        if frame_id is None:
            normalized_frame_id = default_frame_id
        else:
            normalized_frame_id = int(frame_id)
            if normalized_frame_id not in frame_time_lookup:
                continue
        time_value = frame_time_lookup[normalized_frame_id]

        name = str(box.get("name", f"box_{index}"))
        visualizer.add_geometry(
            name=f"{name}_{index}",
            geometry=line_set,
            material=box_material,
            time=time_value,
            group="boxes",
        )

        if show_box_names and "name" in box:
            label_pos = np.asarray(center, dtype=np.float32)
            label_text = str(box["name"])
            if sync_labels_with_frames:
                labels_by_frame.setdefault(normalized_frame_id, []).append((label_pos, label_text))
            else:
                visualizer.add_3d_label(label_pos, label_text)

    return labels_by_frame


def _configure_dynamic_labels(
    *,
    visualizer: o3d.visualization.O3DVisualizer,
    selected_frames: Sequence[int],
    frame_time_step: float,
    labels_by_frame: Mapping[int, Sequence[tuple[np.ndarray, str]]],
) -> None:
    if not selected_frames:
        return

    last_frame_id: dict[str, int | None] = {"value": None}

    def _frame_id_for_time(current_time: float) -> int:
        if len(selected_frames) == 1:
            return int(selected_frames[0])
        index = int(round(current_time / frame_time_step))
        index = max(0, min(index, len(selected_frames) - 1))
        return int(selected_frames[index])

    def _apply_labels(frame_id: int) -> None:
        visualizer.clear_3d_labels()
        for position, text in labels_by_frame.get(frame_id, []):
            visualizer.add_3d_label(np.asarray(position, dtype=np.float32), str(text))
        last_frame_id["value"] = frame_id

    def _on_animation_frame(_: o3d.visualization.O3DVisualizer, current_time: float) -> None:
        frame_id = _frame_id_for_time(current_time)
        if last_frame_id["value"] == frame_id:
            return
        _apply_labels(frame_id)

    visualizer.set_on_animation_frame(_on_animation_frame)
    _apply_labels(int(selected_frames[0]))


def _extract_xyz(
    box: Mapping[str, Any],
    *,
    preferred_key: str,
    fallback_keys: tuple[str, str, str],
    positive_only: bool = False,
) -> np.ndarray:
    if preferred_key in box:
        value = np.asarray(box[preferred_key], dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"Box '{preferred_key}' must contain exactly 3 values.")
        xyz = value
    else:
        try:
            xyz = np.asarray([box[fallback_keys[0]], box[fallback_keys[1]], box[fallback_keys[2]]], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                f"Box must define '{preferred_key}' or all of {fallback_keys}."
            ) from exc

    if positive_only and np.any(xyz <= 0):
        raise ValueError(f"Box '{preferred_key}' values must be > 0.")
    return xyz


def _extract_rotation_matrix(box: Mapping[str, Any]) -> np.ndarray:
    if "rotation_matrix" in box:
        matrix = np.asarray(box["rotation_matrix"], dtype=np.float64)
    elif "R" in box:
        matrix = np.asarray(box["R"], dtype=np.float64)
    else:
        yaw = float(box.get("yaw", 0.0))
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("Box rotation must be a 3x3 matrix.")
    return matrix
