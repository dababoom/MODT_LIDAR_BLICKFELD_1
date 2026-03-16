from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import pandas as pd

import data_io


def data_analysis(path: str | Path) -> None:
    # Load a small batch of lidar frames and print summary statistics.
    # frames = data_io.load_lidar_data(path)
    # print(frames.head())
    # print(frames.describe())

    # print("Frame order matches first timestamp order:", is_frame_id_order_matching_first_timestamp(path),)
    # print("CSV schema is consistent across files:", is_csv_schema_consistent(path),)
    # print_total_size_and_file_count(path)

    #print("Data contains empty cells:", has_empty_cells(path),)
    #print("Data contains zero cells:", has_zero_cells(path),)

    # Visualize multiple distributions in one figure.
    # bins is used 
    plot_multiple_histograms(path, columns=["x", "y", "z"], bins=200, clip_percentiles=(0.00, 1.00))
    # plot_multiple_boxplots(path, columns=["x", "y", "z"], clip_percentiles=(0.0, 1.0))
    # Visualize pairwise scatterplots in one figure.
    # plot_xyz_scatterplots(path, max_points=50000000)
    # Analyze timestamp behavior and show diagnostics.
    # analyze_timestamps(path, max_files=None, show_plots=True)
    


def _column_axis_label(column_name: str) -> str:
    if column_name.upper() in {"X", "Y", "Z"}:
        return f"{column_name} (in meter)"
    return column_name


def _histogram_column_axis_label(column_name: str) -> str:
    upper_name = column_name.upper()
    if upper_name in {"X", "Y", "Z"}:
        return f"{upper_name} (in meter)"
    return upper_name


def _iqr_bounds(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return float("-inf"), float("inf")
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def analyze_timestamps(path: str | Path, max_files: int | None = None, timestamp_scale: float = 1e9, show_plots: bool = True) -> dict[str, object]:
    if timestamp_scale <= 0:
        raise ValueError("timestamp_scale must be > 0.")

    frames = data_io.load_lidar_data(path, max_files=max_files)
    normalized_columns = {name.lower(): name for name in frames.columns}
    timestamp_column = normalized_columns.get("timestamp")
    frame_id_column = normalized_columns.get("frame_id")
    if timestamp_column is None or frame_id_column is None:
        raise ValueError("Required columns missing: timestamp and frame_id.")

    timestamps = pd.to_numeric(frames[timestamp_column], errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError("Timestamp column has no numeric values.")

    frames = frames.loc[timestamps.index].copy()
    frames["TIMESTAMP"] = timestamps

    frame_stats = (
        frames.groupby(frame_id_column, sort=True)["TIMESTAMP"]
        .agg(frame_start="min", frame_end="max", point_count="size")
        .reset_index()
        .rename(columns={frame_id_column: "frame_id"})
    )
    frame_stats["frame_start_s"] = frame_stats["frame_start"] / timestamp_scale
    frame_stats["frame_end_s"] = frame_stats["frame_end"] / timestamp_scale
    frame_stats["frame_duration_s"] = frame_stats["frame_end_s"] - frame_stats["frame_start_s"]
    frame_stats["avg_point_interval_s"] = frame_stats["frame_duration_s"] / (frame_stats["point_count"] - 1).replace(0, pd.NA)
    frame_stats["inter_frame_delta_s"] = frame_stats["frame_start_s"].diff()

    inter_frame_delta_s = frame_stats["inter_frame_delta_s"].dropna()
    recording_span_s = (timestamps.max() - timestamps.min()) / timestamp_scale

    duration_low, duration_high = _iqr_bounds(frame_stats["frame_duration_s"].dropna())
    duration_outlier_mask = frame_stats["frame_duration_s"].lt(duration_low) | frame_stats["frame_duration_s"].gt(duration_high)
    duration_outlier_frames = frame_stats.loc[duration_outlier_mask, "frame_id"].tolist()

    gap_low, gap_high = _iqr_bounds(inter_frame_delta_s)
    gap_outlier_mask = frame_stats["inter_frame_delta_s"].lt(gap_low) | frame_stats["inter_frame_delta_s"].gt(gap_high)
    gap_outlier_frames = frame_stats.loc[gap_outlier_mask, "frame_id"].tolist()

    backward_steps = int((timestamps.diff().dropna() < 0).sum())
    duplicate_timestamps = int(timestamps.duplicated().sum())
    frame_rate_hz = None
    if not inter_frame_delta_s.empty and inter_frame_delta_s.median() > 0:
        frame_rate_hz = 1.0 / inter_frame_delta_s.median()

    print("Timestamp analysis")
    print(f"timestamp_min_raw: {timestamps.min():.0f}")
    print(f"timestamp_max_raw: {timestamps.max():.0f}")
    print(f"recording_span_s: {recording_span_s:.6f}")
    print(f"total_frames: {len(frame_stats)}")
    print(f"monotonic_increasing: {timestamps.is_monotonic_increasing}")
    print(f"backward_steps: {backward_steps}")
    print(f"duplicate_timestamps: {duplicate_timestamps}")
    print(f"frame_duration_mean_s: {frame_stats['frame_duration_s'].mean():.6f}")
    print(f"frame_duration_median_s: {frame_stats['frame_duration_s'].median():.6f}")
    print(f"frame_duration_p95_s: {frame_stats['frame_duration_s'].quantile(0.95):.6f}")
    if not inter_frame_delta_s.empty:
        print(f"inter_frame_mean_s: {inter_frame_delta_s.mean():.6f}")
        print(f"inter_frame_median_s: {inter_frame_delta_s.median():.6f}")
        print(f"inter_frame_std_s: {inter_frame_delta_s.std():.6f}")
        print(f"inter_frame_p95_s: {inter_frame_delta_s.quantile(0.95):.6f}")
        print(f"inter_frame_max_s: {inter_frame_delta_s.max():.6f}")
    print(f"estimated_frame_rate_hz: {frame_rate_hz:.3f}" if frame_rate_hz is not None else "estimated_frame_rate_hz: n/a")
    print(f"duration_outlier_frames_iqr: {len(duration_outlier_frames)}")
    print(f"gap_outlier_frames_iqr: {len(gap_outlier_frames)}")

    if show_plots:
        plot_timestamp_diagnostics(frame_stats)

    return {
        "frame_stats": frame_stats,
        "recording_span_s": recording_span_s,
        "frame_rate_hz": frame_rate_hz,
        "duration_outlier_frame_ids": duration_outlier_frames,
        "gap_outlier_frame_ids": gap_outlier_frames,
        "backward_steps": backward_steps,
        "duplicate_timestamps": duplicate_timestamps,
    }


def plot_timestamp_diagnostics(frame_stats: pd.DataFrame) -> None:
    inter_frame_delta_s = frame_stats["inter_frame_delta_s"].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(frame_stats["frame_id"], frame_stats["frame_start_s"], marker=".", markersize=3, linewidth=1)
    axes[0, 0].set_title("Frame Start Time by Frame ID")
    axes[0, 0].set_xlabel("frame_id")
    axes[0, 0].set_ylabel("frame_start (s)")

    frame_duration_s = frame_stats["frame_duration_s"].dropna()
    frame_duration_mean = frame_duration_s.mean()
    frame_duration_median = frame_duration_s.median()
    frame_duration_min = frame_duration_s.min()
    frame_duration_max = frame_duration_s.max()

    axes[0, 1].hist(frame_duration_s, bins=40, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(frame_duration_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {frame_duration_mean:.7f}")
    axes[0, 1].axvline(frame_duration_median, color="orange", linestyle=":", linewidth=2, label=f"Median: {frame_duration_median:.7f}")
    axes[0, 1].axvline(frame_duration_min, color="green", linestyle="-.", linewidth=2, label=f"Min: {frame_duration_min:.7f}")
    axes[0, 1].axvline(frame_duration_max, color="blue", linestyle="-.", linewidth=2, label=f"Max: {frame_duration_max:.7f}")
    axes[0, 1].set_title("Frame Duration Distribution")
    axes[0, 1].set_xlabel("frame_duration (s)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].xaxis.set_major_formatter(FormatStrFormatter("%.7f"))
    axes[0, 1].tick_params(axis="x", labelrotation=90)
    axes[0, 1].legend(fontsize=8)

    if inter_frame_delta_s.empty:
        axes[1, 0].text(0.5, 0.5, "No inter-frame deltas available", ha="center", va="center")
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()
    else:
        inter_frame_mean = inter_frame_delta_s.mean()
        inter_frame_median = inter_frame_delta_s.median()
        inter_frame_min = inter_frame_delta_s.min()
        inter_frame_max = inter_frame_delta_s.max()

        axes[1, 0].hist(inter_frame_delta_s, bins=40, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(inter_frame_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {inter_frame_mean:.7f}")
        axes[1, 0].axvline(inter_frame_median, color="orange", linestyle=":", linewidth=2, label=f"Median: {inter_frame_median:.7f}")
        axes[1, 0].axvline(inter_frame_min, color="green", linestyle="-.", linewidth=2, label=f"Min: {inter_frame_min:.7f}")
        axes[1, 0].axvline(inter_frame_max, color="blue", linestyle="-.", linewidth=2, label=f"Max: {inter_frame_max:.7f}")
        axes[1, 0].set_title("Inter-Frame Delta Distribution")
        axes[1, 0].set_xlabel("inter_frame_delta (s)")
        axes[1, 0].set_ylabel("count")
        axes[1, 0].xaxis.set_major_formatter(FormatStrFormatter("%.7f"))
        axes[1, 0].tick_params(axis="x", labelrotation=90)
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].boxplot(inter_frame_delta_s, widths=0.5, showfliers=True)
        axes[1, 1].axhline(inter_frame_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {inter_frame_mean:.7f}")
        axes[1, 1].axhline(inter_frame_median, color="orange", linestyle=":", linewidth=2, label=f"Median: {inter_frame_median:.7f}")
        axes[1, 1].axhline(inter_frame_min, color="green", linestyle="-.", linewidth=2, label=f"Min: {inter_frame_min:.7f}")
        axes[1, 1].axhline(inter_frame_max, color="blue", linestyle="-.", linewidth=2, label=f"Max: {inter_frame_max:.7f}")
        axes[1, 1].set_title("Inter-Frame Delta Boxplot")
        axes[1, 1].set_ylabel("inter_frame_delta (s)")
        axes[1, 1].set_xticks([1])
        axes[1, 1].set_xticklabels(["delta"])
        axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_column_histogram(path: str | Path, column: str, max_files: int | None = None, bins: int = 40, clip_percentiles: tuple[float, float] | None = (0.01, 0.99)) -> None:
    frames = data_io.load_lidar_data(path, max_files=max_files)
    normalized_columns = {name.lower(): name for name in frames.columns}
    column_name = normalized_columns.get(column.lower(), column)
    if column_name not in frames.columns:
        raise ValueError(f"Unknown column: {column}")

    values = frames[column_name].dropna()

    plot_values = values
    if clip_percentiles is not None:
        low_q, high_q = clip_percentiles
        if not 0 <= low_q < high_q <= 1:
            raise ValueError("clip_percentiles must be within [0, 1] and low < high.")
        low, high = values.quantile([low_q, high_q])
        plot_values = values[values.between(low, high)]
        if plot_values.empty:
            raise ValueError("No values left after outlier clipping.")

    mean_value = plot_values.mean()
    median_value = plot_values.median()
    min_value = plot_values.min()
    max_value = plot_values.max()

    plt.hist(plot_values, bins=bins, alpha=0.7, edgecolor="black")
    plt.axvline(mean_value, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.3f}")
    plt.axvline(median_value, color="orange", linestyle=":", linewidth=2, label=f"Median: {median_value:.3f}")
    plt.axvline(min_value, color="green", linestyle="-.", linewidth=2, label=f"Min: {min_value:.3f}")
    plt.axvline(max_value, color="blue", linestyle="-.", linewidth=2, label=f"Max: {max_value:.3f}")
    plt.title(f"Histogram: {column_name}")
    plt.xlabel(_histogram_column_axis_label(column_name))
    plt.ylabel("Count (1e6)")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}"))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_histograms(path: str | Path, columns: list[str], max_files: int | None = None, bins: int = 40, clip_percentiles: tuple[float, float] | None = (0.01, 0.99)) -> None:
    if not columns:
        raise ValueError("columns must not be empty.")

    frames = data_io.load_lidar_data(path, max_files=max_files)
    normalized_columns = {name.lower(): name for name in frames.columns}
    resolved_columns = [normalized_columns.get(name.lower(), name) for name in columns]
    for original_name, resolved_name in zip(columns, resolved_columns):
        if resolved_name not in frames.columns:
            raise ValueError(f"Unknown column: {original_name}")

    n_cols = 3 if len(resolved_columns) > 1 else 1
    n_rows = (len(resolved_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, column_name in zip(axes, resolved_columns):
        values = frames[column_name].dropna()
        if values.empty:
            raise ValueError(f"Column '{column_name}' has no values to plot.")

        plot_values = values
        if clip_percentiles is not None:
            low_q, high_q = clip_percentiles
            if not 0 <= low_q < high_q <= 1:
                raise ValueError("clip_percentiles must be within [0, 1] and low < high.")
            low, high = values.quantile([low_q, high_q])
            plot_values = values[values.between(low, high)]
            if plot_values.empty:
                raise ValueError(f"No values left for '{column_name}' after outlier clipping.")

        mean_value = plot_values.mean()
        median_value = plot_values.median()
        min_value = plot_values.min()
        max_value = plot_values.max()

        ax.hist(plot_values, bins=bins, alpha=0.7, edgecolor="black")
        ax.axvline(mean_value, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.3f}")
        ax.axvline(median_value, color="orange", linestyle=":", linewidth=2, label=f"Median: {median_value:.3f}")
        ax.axvline(min_value, color="green", linestyle="-.", linewidth=2, label=f"Min: {min_value:.3f}")
        ax.axvline(max_value, color="blue", linestyle="-.", linewidth=2, label=f"Max: {max_value:.3f}")
        ax.set_title(f"Histogram: {column_name}")
        ax.set_xlabel(_histogram_column_axis_label(column_name))
        ax.set_ylabel("Count (1e6)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1_000_000:g}"))
        ax.legend(fontsize=8)

    for ax in axes[len(resolved_columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_column_boxplot(path: str | Path, column: str, max_files: int | None = None, clip_percentiles: tuple[float, float] | None = (0.01, 0.99)) -> None:
    frames = data_io.load_lidar_data(path, max_files=max_files)
    normalized_columns = {name.lower(): name for name in frames.columns}
    column_name = normalized_columns.get(column.lower(), column)
    if column_name not in frames.columns:
        raise ValueError(f"Unknown column: {column}")

    values = frames[column_name].dropna()
    if values.empty:
        raise ValueError(f"Column '{column_name}' has no values to plot.")

    _ = clip_percentiles
    mean_value = values.mean()
    median_value = values.median()
    min_value = values.min()
    max_value = values.max()

    plt.boxplot(values, widths=0.5, showfliers=True)
    plt.axhline(mean_value, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.3f}")
    plt.axhline(median_value, color="orange", linestyle=":", linewidth=2, label=f"Median: {median_value:.3f}")
    plt.axhline(min_value, color="green", linestyle="-.", linewidth=2, label=f"Min: {min_value:.3f}")
    plt.axhline(max_value, color="blue", linestyle="-.", linewidth=2, label=f"Max: {max_value:.3f}")
    plt.title(f"Boxplot: {column_name}")
    plt.xticks([1], [column_name])
    plt.ylabel(_column_axis_label(column_name))
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_boxplots(
    path: str | Path | pd.DataFrame,
    columns: list[str],
    max_files: int | None = None,
    clip_percentiles: tuple[float, float] | None = (0.01, 0.99),
) -> None:
    if not columns:
        raise ValueError("columns must not be empty.")

    if isinstance(path, pd.DataFrame):
        frames = path.copy()
    else:
        frames = data_io.load_lidar_data(path, max_files=max_files)
    normalized_columns = {name.lower(): name for name in frames.columns}
    resolved_columns = [normalized_columns.get(name.lower(), name) for name in columns]
    for original_name, resolved_name in zip(columns, resolved_columns):
        if resolved_name not in frames.columns:
            raise ValueError(f"Unknown column: {original_name}")

    n_cols = 3 if len(resolved_columns) > 1 else 1
    n_rows = (len(resolved_columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, column_name in zip(axes, resolved_columns):
        values = frames[column_name].dropna()
        if values.empty:
            raise ValueError(f"Column '{column_name}' has no values to plot.")

        plot_values = values
        if clip_percentiles is not None:
            low_q, high_q = clip_percentiles
            if not 0 <= low_q < high_q <= 1:
                raise ValueError("clip_percentiles must be within [0, 1] and low < high.")
            low, high = values.quantile([low_q, high_q])
            plot_values = values[values.between(low, high)]
            if plot_values.empty:
                raise ValueError(f"No values left for '{column_name}' after outlier clipping.")

        mean_value = plot_values.mean()
        median_value = plot_values.median()
        min_value = plot_values.min()
        max_value = plot_values.max()

        ax.boxplot(plot_values, widths=0.5, showfliers=False)
        ax.axhline(mean_value, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_value:.3f}")
        ax.axhline(median_value, color="orange", linestyle=":", linewidth=2, label=f"Median: {median_value:.3f}")
        ax.axhline(min_value, color="green", linestyle="-.", linewidth=2, label=f"Min: {min_value:.3f}")
        ax.axhline(max_value, color="blue", linestyle="-.", linewidth=2, label=f"Max: {max_value:.3f}")
        ax.set_title(f"Boxplot: {column_name}")
        ax.set_xticks([1])
        ax.set_xticklabels([column_name])
        ax.set_ylabel(_column_axis_label(column_name))
        ax.legend(fontsize=8)

    for ax in axes[len(resolved_columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_xyz_scatterplots(path: str | Path, max_files: int | None = None, max_points: int | None = 50000) -> None:
    frames = data_io.load_lidar_data(path, max_files=max_files)
    normalized_columns = {name.lower(): name for name in frames.columns}
    pairs = [("x", "y"), ("y", "z"), ("x", "z")]
    resolved_pairs = [(normalized_columns.get(x, x), normalized_columns.get(y, y)) for x, y in pairs]

    for x_name, y_name in resolved_pairs:
        if x_name not in frames.columns or y_name not in frames.columns:
            raise ValueError(f"Unknown pair: {x_name}, {y_name}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (x_name, y_name) in zip(axes, resolved_pairs):
        points = frames[[x_name, y_name]].dropna()
        if points.empty:
            raise ValueError(f"No values to plot for pair: {x_name}, {y_name}")
        if max_points is not None and len(points) > max_points:
            points = points.sample(n=max_points, random_state=42)

        ax.scatter(points[x_name], points[y_name], s=1, alpha=0.2)
        ax.set_title(f"{x_name} vs {y_name}")
        ax.set_xlabel(_column_axis_label(x_name))
        ax.set_ylabel(_column_axis_label(y_name))

    plt.tight_layout()
    plt.show()


def print_total_size_and_file_count(path: str | Path) -> None:
    files = data_io.list_lidar_frames(path)
    total_bytes = sum(f.stat().st_size for f in files)
    total_points = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            next(file, None)  # skip header
            total_points += sum(1 for _ in file)
    print(f"File count: {len(files)}")
    print(f"Total size: {total_bytes} bytes")
    print(f"Total size: {total_bytes / (1000 ** 3):.3f} GB")
    print(f"Total points: {total_points}")


def is_frame_id_order_matching_first_timestamp(path: str | Path, max_files: int | None = None) -> bool:
    frames = data_io.load_lidar_data(path, max_files=max_files)
    first_timestamps = frames.groupby("frame_id", sort=True)["TIMESTAMP"].first()
    return first_timestamps.is_monotonic_increasing


def is_csv_schema_consistent(path: str | Path, max_files: int | None = None) -> bool:
    files = data_io.list_lidar_frames(path)
    if max_files is not None:
        files = files[:max_files]

    expected_schema: tuple[str, ...] | None = None
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            schema = tuple(file.readline().strip().split(";"))
        if expected_schema is None:
            expected_schema = schema
            continue
        if schema != expected_schema:
            return False
    return True


def has_empty_cells(path: str | Path, max_files: int | None = None) -> bool:
    frames = data_io.load_lidar_data(path, max_files=max_files)
    # checks whether frames contains at least one missing value (NaN/None) anywhere
    return frames.isna().any().any()


def has_zero_cells(path: str | Path, max_files: int | None = None) -> bool:
    frames = data_io.load_lidar_data(path, max_files=max_files)
    frames = frames.drop(columns=["RETURN_ID"], errors="ignore")
    # checks whether frames contains at least one cell with value 0 anywhere
    zero_mask = frames.eq(0)
    has_zero = zero_mask.any().any()

    if has_zero:
        zero_locations = zero_mask.stack()
        zero_locations = zero_locations[zero_locations]
        for row_index, column_name in zero_locations.index:
            frame_id_value = frames.at[row_index, "frame_id"] if "frame_id" in frames.columns else "unknown"
            print(f"Zero detected: frame_id={frame_id_value}, column={column_name}")

    return has_zero
