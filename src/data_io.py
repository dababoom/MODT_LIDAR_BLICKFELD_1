from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import open3d as o3d
import pandas as pd


# Accept both raw-file naming (`..._frame-123.csv`) and saved-frame naming (`frame-123.csv`).
FRAME_RE = re.compile(r"(?:^|[_-])frame[-_](\d+)\.csv$", re.IGNORECASE)
SAVED_FRAME_RE = re.compile(r"^frame-(\d+)\.csv$", re.IGNORECASE)
DEFAULT_REQUIRED_FRAME_COLUMNS = ("x", "y", "z")
DEFAULT_OPTIONAL_FRAME_COLUMNS = ("intensity", "timestamp")
DEFAULT_ORDERED_FRAME_COLUMNS = ("frame_id", "timestamp", "intensity", "x", "y", "z")


def frame_id_from_filename(file_path: str | Path, *, strict: bool = False) -> int:
    """Extract the numeric frame id from a raw or saved LiDAR CSV file name."""
    path = Path(file_path)
    pattern = SAVED_FRAME_RE if strict else FRAME_RE
    match = pattern.match(path.name) if strict else pattern.search(path.name)
    if match is None:
        expected_pattern = "'frame-<id>.csv'" if strict else "a frame-based CSV name"
        raise ValueError(f"CSV filename does not match {expected_pattern}: {path.name}")
    return int(match.group(1))


def list_lidar_frames(raw_dir: str | Path) -> list[Path]:
    """Return raw LiDAR CSV files sorted by their frame id."""
    raw_path = Path(raw_dir)
    files = sorted(raw_path.rglob("*.csv"), key=frame_id_from_filename)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {raw_path}")
    return files


def list_saved_frame_files(frame_dir: str | Path, max_files: int | None = None) -> list[Path]:
    """Return saved `frame-<id>.csv` files sorted by frame id.

    Files with suffixes such as `frame-123 2.csv` are intentionally ignored so
    downstream stages only consume the canonical pipeline outputs.
    """
    directory = Path(frame_dir)
    files = sorted(
        (path for path in directory.glob("frame-*.csv") if SAVED_FRAME_RE.match(path.name)),
        key=lambda path: frame_id_from_filename(path, strict=True),
    )
    if max_files is not None:
        files = files[:max_files]
    return files


def load_lidar_data(raw_dir: str | Path = "data/LidarData/raw", max_files: int | None = None) -> pd.DataFrame:
    """Load raw LiDAR CSV files and append their originating frame id."""
    files = list_lidar_frames(raw_dir)
    if max_files is not None:
        files = files[:max_files]

    frames: list[pd.DataFrame] = []
    for file_path in files:
        frame_id = frame_id_from_filename(file_path)
        frame_table = pd.read_csv(file_path, sep=";")
        frame_table["frame_id"] = frame_id
        frames.append(frame_table)

    return pd.concat(frames, ignore_index=True)


def normalize_frame_table(
    frame_table: pd.DataFrame,
    *,
    frame_id: int,
    required_columns: tuple[str, ...] = DEFAULT_REQUIRED_FRAME_COLUMNS,
    optional_columns: tuple[str, ...] = DEFAULT_OPTIONAL_FRAME_COLUMNS,
) -> pd.DataFrame:
    """Rename a frame table to the canonical pipeline schema and coerce numerics."""
    normalized_columns = {column.lower(): column for column in frame_table.columns}
    missing_columns = [column_name for column_name in required_columns if column_name not in normalized_columns]
    if missing_columns:
        raise ValueError(f"Required columns missing for frame {frame_id}: {missing_columns}")

    rename_map = {normalized_columns[column_name]: column_name for column_name in required_columns}
    for optional_column in optional_columns:
        optional_original = normalized_columns.get(optional_column)
        if optional_original is not None:
            rename_map[optional_original] = optional_column
    if "frame_id" in normalized_columns:
        rename_map[normalized_columns["frame_id"]] = "frame_id"

    normalized_table = frame_table.rename(columns=rename_map).copy()

    numeric_columns = [*required_columns]
    numeric_columns.extend(column_name for column_name in optional_columns if column_name in normalized_table.columns)
    if "frame_id" in normalized_table.columns:
        numeric_columns.append("frame_id")
    for column_name in numeric_columns:
        normalized_table[column_name] = pd.to_numeric(normalized_table[column_name], errors="coerce")

    normalized_table = normalized_table.dropna(subset=list(required_columns)).reset_index(drop=True)
    normalized_table["frame_id"] = int(frame_id)

    ordered_columns = [column_name for column_name in DEFAULT_ORDERED_FRAME_COLUMNS if column_name in normalized_table.columns]
    remaining_columns = [column_name for column_name in normalized_table.columns if column_name not in ordered_columns]
    return normalized_table[ordered_columns + remaining_columns]


def load_saved_frame_table(
    frame_file: str | Path,
    *,
    separator: str = ";",
    required_columns: tuple[str, ...] = DEFAULT_REQUIRED_FRAME_COLUMNS,
    optional_columns: tuple[str, ...] = DEFAULT_OPTIONAL_FRAME_COLUMNS,
) -> tuple[int, pd.DataFrame]:
    """Load one saved frame CSV into the canonical pipeline table format."""
    frame_path = Path(frame_file)
    frame_id = frame_id_from_filename(frame_path, strict=True)
    frame_table = pd.read_csv(frame_path, sep=separator)
    normalized_table = normalize_frame_table(
        frame_table,
        frame_id=frame_id,
        required_columns=required_columns,
        optional_columns=optional_columns,
    )
    return frame_id, normalized_table


def load_saved_frame_tables(
    frame_dir: str | Path,
    *,
    frame_ids: list[int] | None = None,
    separator: str = ";",
    max_files: int | None = None,
    required_columns: tuple[str, ...] = DEFAULT_REQUIRED_FRAME_COLUMNS,
    optional_columns: tuple[str, ...] = DEFAULT_OPTIONAL_FRAME_COLUMNS,
) -> pd.DataFrame:
    """Load multiple saved frame CSVs and concatenate the valid tables."""
    directory = Path(frame_dir)
    if not directory.exists():
        return pd.DataFrame(columns=list(DEFAULT_ORDERED_FRAME_COLUMNS))

    selected_frame_ids = None if frame_ids is None else {int(frame_id) for frame_id in frame_ids}
    frame_files = list_saved_frame_files(directory, max_files=max_files)
    if selected_frame_ids is not None:
        frame_files = [
            file_path
            for file_path in frame_files
            if frame_id_from_filename(file_path, strict=True) in selected_frame_ids
        ]

    frames: list[pd.DataFrame] = []
    for file_path in frame_files:
        try:
            _, frame_table = load_saved_frame_table(
                file_path,
                separator=separator,
                required_columns=required_columns,
                optional_columns=optional_columns,
            )
        except ValueError:
            continue
        frames.append(frame_table)

    if not frames:
        return pd.DataFrame(columns=list(DEFAULT_ORDERED_FRAME_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def convert_pd_to_point_cloud(frames: pd.DataFrame) -> o3d.t.geometry.PointCloud:
    """Convert a canonical LiDAR table into an Open3D tensor point cloud."""
    normalized_columns = {name.lower(): name for name in frames.columns}
    required = ["x", "y", "z", "intensity", "timestamp"]
    missing_columns = [name for name in required if name not in normalized_columns]
    if missing_columns:
        raise ValueError(f"Required columns missing for point cloud conversion: {missing_columns}")

    x_column, y_column, z_column, intensity_column, timestamp_column = [normalized_columns[name] for name in required]
    frame_column = normalized_columns.get("frame_id")
    selected_columns = [x_column, y_column, z_column, intensity_column, timestamp_column]
    if frame_column is not None:
        selected_columns.append(frame_column)

    points = frames[selected_columns].copy()
    for column in selected_columns:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points = points.dropna()
    if (points[intensity_column] < 0).any():
        raise ValueError("Intensity values must be >= 0 for UInt32 conversion.")
    if (points[timestamp_column] < 0).any():
        raise ValueError("Timestamp values must be >= 0 for UInt64 conversion.")

    positions = points[[x_column, y_column, z_column]].astype("float32").to_numpy()
    intensities = points[[intensity_column]].astype("uint32").to_numpy()
    timestamps = points[[timestamp_column]].astype("uint64").to_numpy()
    frame_ids = points[[frame_column]].astype("int64").to_numpy() if frame_column is not None else None

    point_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(positions, dtype=o3d.core.Dtype.Float32))
    point_cloud.point.intensity = o3d.core.Tensor(intensities, dtype=o3d.core.Dtype.UInt32)
    point_cloud.point.timestamp = o3d.core.Tensor(timestamps, dtype=o3d.core.Dtype.UInt64)
    if frame_ids is not None:
        point_cloud.point.frame_id = o3d.core.Tensor(frame_ids, dtype=o3d.core.Dtype.Int64)
    return point_cloud


def point_cloud_to_dataframe(point_cloud: o3d.t.geometry.PointCloud) -> pd.DataFrame:
    """Convert an Open3D tensor point cloud into a flat attribute table."""
    column_data: dict[str, np.ndarray] = {}
    for attribute_name in list(point_cloud.point):
        attribute_values = point_cloud.point[attribute_name].numpy()
        attribute_matrix = (
            attribute_values.reshape(-1, 1)
            if attribute_values.ndim == 1
            else attribute_values.reshape(attribute_values.shape[0], -1)
        )
        if attribute_name == "positions":
            axis_names = ["x", "y", "z"]
            for column_index in range(attribute_matrix.shape[1]):
                column_name = axis_names[column_index] if column_index < 3 else f"positions_{column_index}"
                column_data[column_name] = attribute_matrix[:, column_index]
            continue
        if attribute_matrix.shape[1] == 1:
            column_data[attribute_name] = attribute_matrix[:, 0]
            continue
        for column_index in range(attribute_matrix.shape[1]):
            column_data[f"{attribute_name}_{column_index}"] = attribute_matrix[:, column_index]

    point_table = pd.DataFrame(column_data)
    if "frame_id" in point_table.columns:
        point_table["frame_id"] = point_table["frame_id"].astype(np.int64)
    return point_table


def save_point_cloud_frames_to_csv(
    point_cloud: o3d.t.geometry.PointCloud,
    output_dir: str | Path = "data/preprocessed",
    file_prefix: str = "frame",
) -> list[Path]:
    """Save a tensor point cloud as one canonical CSV file per frame."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for existing_file in output_path.glob(f"{file_prefix}-*.csv"):
        existing_file.unlink()

    point_table = point_cloud_to_dataframe(point_cloud)
    if point_table.empty:
        return []
    if "frame_id" in point_table.columns:
        frame_ids = sorted(point_table["frame_id"].astype(np.int64).unique().tolist())
    else:
        frame_ids = [0]
        point_table["frame_id"] = 0

    written_files: list[Path] = []
    for frame_id in frame_ids:
        frame_table = point_table.loc[point_table["frame_id"] == frame_id]
        frame_file = output_path / f"{file_prefix}-{int(frame_id)}.csv"
        frame_table.to_csv(frame_file, sep=";", index=False)
        written_files.append(frame_file)
    return written_files
