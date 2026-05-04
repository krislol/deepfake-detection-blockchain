"""
ML module for deepfake detection.
Provides video processing, face detection, and model inference.
"""

from .processing import (
    extract_frames,
    detect_faces,
    extract_face_regions,
    preprocess_faces,
    aggregate_predictions,
    load_celeb_df_video
)

from .model import (
    DeepfakeDetectionModel,
    DeepfakeDetector,
    FaceDataset,
    get_data_transforms
)

__all__ = [
    # Processing
    'extract_frames',
    'detect_faces',
    'extract_face_regions',
    'preprocess_faces',
    'aggregate_predictions',
    'load_celeb_df_video',
    # Model
    'DeepfakeDetectionModel',
    'DeepfakeDetector',
    'FaceDataset',
    'get_data_transforms',
]

__version__ = '0.1.0'
