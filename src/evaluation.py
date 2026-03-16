from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pipeline_config import MAX_MISSED, MIN_HITS_CONFIRM


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compute_tracking_summary_metrics() -> dict[str, int]:
    measurement_files = sorted((PROJECT_ROOT / "data" / "measurements").glob("frame-*.csv"))
    tracks = pd.read_csv(PROJECT_ROOT / "data" / "tracks" / "tracks.csv", sep=";")

    # Calculate the number of generated measurement clusters from object detection.
    measurement_clusters = sum(sum(1 for _ in measurement_file.open()) - 1 for measurement_file in measurement_files)
    # Calculate the number of measurement clusters that are confirmed in a track.
    confirmed_measurement_clusters = int((tracks["missed_frames"] == 0).sum())
    # Calculate the number of tracks.
    number_of_tracks = int(tracks["track_id"].nunique())
    # Calculate the confirmed measurement clusters plus the minimum confirmation hits that were needed to start each confirmed track.
    confirmed_measurement_clusters_with_confirmation_hits = confirmed_measurement_clusters + number_of_tracks * MIN_HITS_CONFIRM
    print(f"Confirmed measurement clusters plus the minimum confirmation hits needed to start each confirmed track: {confirmed_measurement_clusters_with_confirmation_hits} (should be <= {measurement_clusters})")
    # Calculate the number of saved confirmed JPDA track states, which is equal to the number of green track bounding boxes in the visualization.
    confirmed_jpda_track_states = int(len(tracks))
    # Calculate how many times missed_frames was not 0.
    missing_measurements = int((tracks["missed_frames"] != 0).sum())
    # Calculate the number of missing measurements during active tracks without the missing measurements from track endings.
    missing_measurements_during_active_tracks = missing_measurements - number_of_tracks * MAX_MISSED
    # Calculate the number of JPDA state estimates without feasible measurement, including the allowed track-ending misses.
    jpda_state_estimates_without_feasible_measurement = missing_measurements_during_active_tracks + number_of_tracks * MAX_MISSED
    return {
        "measurement_clusters": measurement_clusters,
        "confirmed_measurement_clusters": confirmed_measurement_clusters,
        "confirmed_jpda_track_states": confirmed_jpda_track_states,
        "missing_measurements_during_active_tracks": missing_measurements_during_active_tracks,
        "jpda_state_estimates_without_feasible_measurement": jpda_state_estimates_without_feasible_measurement,
    }


def _print_tracking_summary(metrics: dict[str, int]) -> None:
    print(f"Generated measurement clusters: {metrics['measurement_clusters']}")
    print(f"Confirmed measurement clusters in tracks: {metrics['confirmed_measurement_clusters']}")
    print(f"Confirmed JPDA track states: {metrics['confirmed_jpda_track_states']}")
    print(f"JPDA state estimates without feasible measurement: {metrics['jpda_state_estimates_without_feasible_measurement']}")
    print(f"Missing measurements during active tracks without track ending misses: {metrics['missing_measurements_during_active_tracks']}")


def _plot_tracking_summary_chart(metrics: dict[str, int]) -> None:
    items = [
        ("measurement_clusters", "Measurements\n(clusters from object detection)"),
        ("confirmed_measurement_clusters", "Measurements used in confirmed tracks"),
        ("confirmed_jpda_track_states", "JPDA state estimates\n(green track boxes in Video 8)"),
        ("jpda_state_estimates_without_feasible_measurement", "JPDA state estimates\nwithout feasible measurement"),
        ("missing_measurements_during_active_tracks", "JPDA state estimates\nwithout feasible measurement\n(excluding track terminations)"),
    ]
    fig, ax = plt.subplots(figsize=(20, 4.8))
    bars = ax.bar([label for _, label in items], [metrics[key] for key, _ in items], alpha=0.7, edgecolor="black")
    for bar, (key, _) in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{metrics[key]}", ha="center", va="bottom")
    ax.set_title("Object Detection and Tracking Summary")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", labelrotation=0)
    plt.tight_layout()
    plt.show()


def tracking_summary() -> None:
    metrics = _compute_tracking_summary_metrics()
    _print_tracking_summary(metrics)
    _plot_tracking_summary_chart(metrics)


def _summary_stats(values: pd.Series) -> tuple[float, float, float, float]:
    return float(values.mean()), float(values.median()), float(values.min()), float(values.max())


def _add_summary_legend(ax: plt.Axes, stats: tuple[float, float, float, float] | None = None) -> None:
    if stats is None:
        labels = ("Mean", "Median", "Min", "Max")
    else:
        mean_value, median_value, min_value, max_value = stats
        labels = (
            f"Mean: {mean_value:.3f}",
            f"Median: {median_value:.3f}",
            f"Min: {min_value:.3f}",
            f"Max: {max_value:.3f}",
        )
    ax.plot([], [], color="red", linestyle="--", linewidth=2, label=labels[0])
    ax.plot([], [], color="orange", linestyle=":", linewidth=2, label=labels[1])
    ax.plot([], [], color="green", linestyle="-.", linewidth=2, label=labels[2])
    ax.plot([], [], color="blue", linestyle="-.", linewidth=2, label=labels[3])
    ax.legend(fontsize=8)


def _add_vertical_summary_lines(ax: plt.Axes, values: pd.Series) -> None:
    mean_value, median_value, min_value, max_value = _summary_stats(values)
    ax.axvline(mean_value, color="red", linestyle="--", linewidth=2)
    ax.axvline(median_value, color="orange", linestyle=":", linewidth=2)
    ax.axvline(min_value, color="green", linestyle="-.", linewidth=2)
    ax.axvline(max_value, color="blue", linestyle="-.", linewidth=2)
    _add_summary_legend(ax, (mean_value, median_value, min_value, max_value))


def _add_horizontal_summary_lines(ax: plt.Axes, values: pd.Series) -> None:
    mean_value, median_value, min_value, max_value = _summary_stats(values)
    ax.axhline(mean_value, color="red", linestyle="--", linewidth=2)
    ax.axhline(median_value, color="orange", linestyle=":", linewidth=2)
    ax.axhline(min_value, color="green", linestyle="-.", linewidth=2)
    ax.axhline(max_value, color="blue", linestyle="-.", linewidth=2)
    _add_summary_legend(ax, (mean_value, median_value, min_value, max_value))


def _add_boxplot_summary_segments(ax: plt.Axes, values: pd.Series, x_center: float, half_width: float = 0.25) -> None:
    mean_value, median_value, min_value, max_value = _summary_stats(values)
    ax.hlines(mean_value, x_center - half_width, x_center + half_width, colors="red", linestyles="--", linewidth=2)
    ax.hlines(median_value, x_center - half_width, x_center + half_width, colors="orange", linestyles=":", linewidth=2)
    ax.hlines(min_value, x_center - half_width, x_center + half_width, colors="green", linestyles="-.", linewidth=2)
    ax.hlines(max_value, x_center - half_width, x_center + half_width, colors="blue", linestyles="-.", linewidth=2)


def _add_horizontal_summary_segments(ax: plt.Axes, values: pd.Series, x_min: float, x_max: float) -> None:
    mean_value, median_value, min_value, max_value = _summary_stats(values)
    ax.hlines(mean_value, x_min, x_max, colors="red", linestyles="--", linewidth=2)
    ax.hlines(median_value, x_min, x_max, colors="orange", linestyles=":", linewidth=2)
    ax.hlines(min_value, x_min, x_max, colors="green", linestyles="-.", linewidth=2)
    ax.hlines(max_value, x_min, x_max, colors="blue", linestyles="-.", linewidth=2)
    _add_summary_legend(ax, (mean_value, median_value, min_value, max_value))


def _load_valid_track_states() -> tuple[pd.DataFrame, pd.Series, list[int]]:
    tracks = pd.read_csv(PROJECT_ROOT / "data" / "tracks" / "tracks.csv", sep=";")
    last_classes = tracks.sort_values("frame_id").groupby("track_id")["classification_label"].last()
    tracks = tracks.loc[tracks["missed_frames"] == 0].copy()
    track_ids = sorted(tracks["track_id"].value_counts().loc[lambda counts: counts > 5].index.tolist())
    tracks = tracks.loc[tracks["track_id"].isin(track_ids)].copy()
    tracks["posterior_residual_m"] = np.sqrt(
        (tracks["x"] - tracks["measurement_centroid_x"]) ** 2
        + (tracks["y"] - tracks["measurement_centroid_y"]) ** 2
        + (tracks["z"] - tracks["measurement_centroid_z"]) ** 2
    )
    return tracks, last_classes, track_ids


def plot_track_bbox_volumes() -> None:
    tracks = pd.read_csv(PROJECT_ROOT / "data" / "tracks" / "tracks.csv", sep=";")
    last_classes = tracks.sort_values("frame_id").groupby("track_id")["classification_label"].last()
    tracks = tracks.loc[tracks["missed_frames"] == 0].copy()
    track_ids = sorted(tracks["track_id"].value_counts().loc[lambda counts: counts > 5].index.tolist())
    fig, axes = plt.subplots(len(track_ids), 1, figsize=(12, 3 * len(track_ids)), sharex=True)
    if len(track_ids) == 1:
        axes = [axes]
    for ax, track_id in zip(axes, track_ids):
        track = tracks.loc[tracks["track_id"] == track_id].sort_values("frame_id")
        ax.plot(track["frame_id"], track["bbox_volume"])
        _add_horizontal_summary_segments(ax, track["bbox_volume"], float(track["frame_id"].min()), float(track["frame_id"].max()))
        ax.set_title(f"Track {track_id}: {last_classes.loc[track_id]}")
    fig.supylabel("bbox_volume [m^3]")
    fig.supxlabel("frame_id")
    fig.suptitle("Measured bbox_volume [m^3] over frame")
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.50, bottom=0.08, top=0.907)
    plt.show()


def plot_jpda_posterior_statistics() -> None:
    tracks, _, _ = _load_valid_track_states()
    print(f"Mean JPDA posterior covariance x [m^2]: {tracks['cov_x'].mean():.4f}")
    print(f"Mean JPDA posterior covariance y [m^2]: {tracks['cov_y'].mean():.4f}")
    print(f"Mean JPDA posterior covariance z [m^2]: {tracks['cov_z'].mean():.4f}")
    print(f"Mean posterior residual [m]: {tracks['posterior_residual_m'].mean():.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].boxplot([tracks["cov_x"], tracks["cov_y"], tracks["cov_z"]], labels=["cov_x", "cov_y", "cov_z"])
    axes[0].set_title("JPDA posterior position variances for all tracks")
    axes[0].set_ylabel("variance [m^2]")
    axes[1].boxplot([tracks["posterior_residual_m"]], labels=["residual"])
    _add_horizontal_summary_lines(axes[1], tracks["posterior_residual_m"])
    axes[1].set_title("Posterior residual distance for all tracks")
    axes[1].set_ylabel("distance [m]")
    plt.tight_layout()
    plt.show()


def plot_track_posterior_residual_histograms() -> None:
    tracks, last_classes, track_ids = _load_valid_track_states()
    bins = np.linspace(tracks["posterior_residual_m"].min(), tracks["posterior_residual_m"].max(), 15)
    fig, axes = plt.subplots(len(track_ids) + 1, 1, figsize=(12, 3 * (len(track_ids) + 1)), sharex=True)
    if len(track_ids) == 0:
        axes = [axes]
    axes[0].hist(tracks["posterior_residual_m"], bins=bins, edgecolor="black")
    _add_vertical_summary_lines(axes[0], tracks["posterior_residual_m"])
    axes[0].set_title("All tracks")
    for ax, track_id in zip(axes[1:], track_ids):
        track = tracks.loc[tracks["track_id"] == track_id].sort_values("frame_id")
        ax.hist(track["posterior_residual_m"], bins=bins, edgecolor="black")
        _add_vertical_summary_lines(ax, track["posterior_residual_m"])
        ax.set_title(f"Track {track_id}: {last_classes.loc[track_id]}")
    fig.supylabel("count")
    fig.supxlabel("posterior residual [m]")
    fig.suptitle("Posterior residual histograms per track")
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.50, bottom=0.08, top=0.907)
    plt.show()


def plot_track_jpda_posterior_boxplots() -> None:
    tracks, last_classes, track_ids = _load_valid_track_states()
    cov_rows = 2
    cov_cols = int(np.ceil(len(track_ids) / cov_rows))
    fig_cov, axes_cov = plt.subplots(cov_rows, cov_cols, figsize=(5 * cov_cols, 4.5 * cov_rows))
    axes_cov = np.atleast_1d(axes_cov).ravel(order="F")
    for ax, track_id in zip(axes_cov, track_ids):
        track = tracks.loc[tracks["track_id"] == track_id].sort_values("frame_id")
        ax.boxplot([track["cov_x"], track["cov_y"], track["cov_z"]], labels=["cov_x", "cov_y", "cov_z"])
        ax.set_title(f"Track {track_id}: {last_classes.loc[track_id]}")
        ax.set_ylabel("variance [m^2]")
    for ax in axes_cov[len(track_ids):]:
        ax.set_visible(False)
    fig_cov.suptitle("JPDA posterior position variances per track")
    plt.tight_layout()
    fig_cov.subplots_adjust(hspace=0.50, wspace=0.30, bottom=0.06, top=0.9)
    plt.show()
    res_rows = 2
    residual_plot_count = len(track_ids) + 1
    res_cols = int(np.ceil(residual_plot_count / res_rows))
    fig_res, axes_res = plt.subplots(res_rows, res_cols, figsize=(5 * res_cols, 4.5 * res_rows))
    axes_res = np.atleast_1d(axes_res).ravel(order="F")
    axes_res[0].boxplot([tracks["posterior_residual_m"]], labels=["posterior residual"])
    _add_horizontal_summary_lines(axes_res[0], tracks["posterior_residual_m"])
    axes_res[0].set_title("All tracks")
    axes_res[0].set_ylabel("distance [m]")
    for ax, track_id in zip(axes_res[1:], track_ids):
        track = tracks.loc[tracks["track_id"] == track_id].sort_values("frame_id")
        ax.boxplot([track["posterior_residual_m"]], labels=["posterior residual"])
        _add_horizontal_summary_lines(ax, track["posterior_residual_m"])
        ax.set_title(f"Track {track_id}: {last_classes.loc[track_id]}")
        ax.set_ylabel("distance [m]")
    for ax in axes_res[residual_plot_count:]:
        ax.set_visible(False)
    fig_res.suptitle("Posterior residual per track")
    plt.tight_layout()
    fig_res.subplots_adjust(hspace=0.50, wspace=0.30, bottom=0.06, top=0.9)
    plt.show()


def evaluation() -> None:
    # tracking_summary()
    # plot_track_bbox_volumes()
    plot_jpda_posterior_statistics()
    plot_track_jpda_posterior_boxplots()
    plot_track_posterior_residual_histograms()
