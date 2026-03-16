import os
from pathlib import Path

# Used for seed generation mainly for ground segmentation
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from data_analysis import data_analysis
from evaluation import evaluation
from data_preprocessing import data_preprocessing
from object_tracking import JPDATrackingConfig, run_tracking_jpda
from object_detection import MotionDetectionConfig, motion_detection
from data_preprocessing import data_preprocessing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_RAW_DATA = PROJECT_ROOT / "data" / "LidarData" / "raw"
# PATH_RAW_DATA = PROJECT_ROOT / "data" / "LidarData" / "quick_check_sample_30"
# PATH_RAW_DATA = PROJECT_ROOT / "data" / "LidarData" / "example_passenger"
PATH_PREPROCESSED_DATA = PROJECT_ROOT / "data" / "preprocessed"
PATH_MEASUREMENTS = PROJECT_ROOT / "data" / "measurements"


def main() -> None:
    # Run the project's default lidar analysis workflow.
    data_analysis(path=PATH_RAW_DATA)
    
    # Run the project's default lidar preprocessing workflow.
    data_preprocessing(path=PATH_RAW_DATA, visualization=False)
    # data_preprocessing(path=PATH_RAW_DATA, max_files=1000000, visualization=True)
    # data_preprocessing(path=PATH_RAW_DATA, visualization=False)

    # Run motion-based detection on preprocessed frames.
    motion_detection(path=PATH_PREPROCESSED_DATA, config=MotionDetectionConfig(labels_output_dir=PROJECT_ROOT / "data" / "motion_labels", measurements_output_dir=PROJECT_ROOT / "data" / "measurements", clear_outputs=True, visualization=True, visualization_max_frames=3000))
    
    run_tracking_jpda(
        config=JPDATrackingConfig(
            preprocessed_dir=PATH_PREPROCESSED_DATA,
            labels_output_dir=PROJECT_ROOT / "data" / "motion_labels",
            measurements_dir=PROJECT_ROOT / "data" / "measurements",
            tracks_output_file=PROJECT_ROOT / "data" / "tracks" / "tracks.csv",
            visualization=True,
            visualization_max_frames=3000,
            visualization_window_title="JPDA tracking (3D boxes + track IDs)",
        )
    )
    
    evaluation()
    


if __name__ == "__main__":
    main()
