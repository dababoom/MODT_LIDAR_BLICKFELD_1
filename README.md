# LiDAR MODT Pipeline

## Final Results Video
![Final results video](https://github.com/user-attachments/assets/24bb93a6-c016-4c7a-a457-2619ad1789af)


## Project Overview
This repository contains a LiDAR-based perception pipeline for stationary traffic scenes. The pipeline reads raw Blickfeld Cube 1 CSV frames, preprocesses the point clouds, detects dynamic objects, tracks them with JPDA, and evaluates the resulting trajectories.

The implementation is organized as a modular thesis project. The main pipeline entry point is `src/main.py`, while the individual stages remain available as separate modules.

## Pipeline Overview
The default pipeline in `src/main.py` executes the following stages:

1. `data_preprocessing`
   Loads raw LiDAR CSV files, removes outliers, crops the region of interest, removes the ground plane, and writes preprocessed frame files.
2. `object_detection`
   Performs motion-based detection with voxel occupancy history and clustering, then writes motion labels and measurement files.
3. `object_tracking`
   Runs JPDA tracking on the measurement files and writes tracked states to `data/tracks/tracks.csv`.
4. `evaluation`
   Computes summary statistics and plots for the tracking results.

## Repository Structure
```text
src/
  main.py
  pipeline_config.py
  data_io.py
  data_preprocessing.py
  data_analysis.py
  object_detection.py
  object_tracking.py
  evaluation.py
  point_cloud_visualization.py

data/
  LidarData/raw/
  preprocessed/
  motion_labels/
  measurements/
  tracks/
```

## Setup
This project uses `pipenv` and targets Python `3.10.19`.

Install dependencies from the project root:

```bash
pipenv sync
```

## How To Run
Run the full pipeline from the project root:

```bash
pipenv run python src/main.py
```

Run only the analysis stage:

```bash
pipenv run python -c "from data_analysis import data_analysis; from pathlib import Path; data_analysis(Path('data/LidarData/raw'))"
```

Run only preprocessing:

```bash
pipenv run python -c "from data_preprocessing import data_preprocessing; from pathlib import Path; data_preprocessing(Path('data/LidarData/raw'))"
```

## Configuration
Pipeline hyperparameters are defined in `src/pipeline_config.py`.

They are grouped into:

- `PreprocessingHyperparameters`
- `DetectionHyperparameters`
- `TrackingHyperparameters`

The main pipeline uses these parameter groups without removing stage-specific entry points. The existing parameters should therefore be treated as part of the public pipeline interface.

## Input/Output
### Input
Raw LiDAR data is expected as CSV files in `data/LidarData/raw/`, recursively organized in folders. File names must contain a frame id such as `..._frame-1849.csv`.

Expected raw schema:

```text
X;Y;Z;DISTANCE;INTENSITY;POINT_ID;RETURN_ID;AMBIENT;TIMESTAMP
```

The pipeline primarily uses:

- `X`, `Y`, `Z`
- `TIMESTAMP`

### Output
#### Preprocessed Frames
Written to `data/preprocessed/frame-<id>.csv`.

Schema:

```text
frame_id;timestamp;intensity;x;y;z
```

#### Motion Labels
Written to `data/motion_labels/frame-<id>.csv`.

Schema:

```text
frame_id;timestamp;intensity;x;y;z;voxel_ix;voxel_iy;voxel_iz;voxel_ratio;motion_class_id;motion_class;motion_cluster_id
```

#### Measurements
Written to `data/measurements/frame-<id>.csv`.

Schema:

```text
frame_id;cluster_id;point_count;bbox_volume;centroid_x;centroid_y;centroid_z;
bbox_min_x;bbox_min_y;bbox_min_z;bbox_max_x;bbox_max_y;bbox_max_z;
bbox_dx;bbox_dy;bbox_dz;
cov_xx;cov_xy;cov_xz;cov_yx;cov_yy;cov_yz;cov_zx;cov_zy;cov_zz;mean_intensity
```

Each row represents one detected moving cluster in a frame.

#### Tracks
Written to `data/tracks/tracks.csv`.

Schema:

```text
frame_id;track_id;x;vx;y;vy;z;vz;speed_mps;missed_frames;
cov_x;cov_y;cov_z;
measurement_centroid_x;measurement_centroid_y;measurement_centroid_z;
bbox_mean_center_x;bbox_mean_center_y;bbox_mean_center_z;
bbox_dx;bbox_dy;bbox_dz;
point_count;bbox_volume;mean_intensity;classification_label
```

Each row represents one confirmed tracked state for one frame.

## Method Summary
- Preprocessing combines statistical outlier removal, radius outlier removal, ROI cropping, and per-frame ground removal.
- Detection uses voxel occupancy over a rolling time window to distinguish static and moving points, then groups moving points with DBSCAN.
- Tracking uses JPDA with a constant-velocity motion model in 3D.
- Classification is currently rule-based and uses boundingbox geometry and distance.

## Limitations
- The pipeline currently targets stationary traffic scenes and should be extended with ego-motion compensation for use on a moving vehicle.
- The main limitation is object detection: it can produce noisy clusters and only detects dynamic objects.
- Tracking is already robust, but nearby objects can still create association ambiguities.
- Wider evaluation on more traffic scenarios is still needed to tune the hyperparameters more broadly.

## AI Assistance
Parts of the development and refactoring process were supported by AI-assisted tools, including OpenAI Codex. Final design decisions, implementation choices, validation, and responsibility for the code remain with the author.
