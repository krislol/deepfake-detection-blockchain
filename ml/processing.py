"""
Video processing module for deepfake detection.
Handles video loading, frame extraction, face detection, and preprocessing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
except (ImportError, AttributeError):
    # Fallback to OpenCV Haar Cascades if MediaPipe fails
    logger.warning("MediaPipe not available, using OpenCV Haar Cascades for face detection")
    mp_face_detection = None
    mp_drawing = None


def extract_frames(video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
    """Extract every nth frame from a video file and resize to 640×480."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video at {video_path}")
        return []

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % sample_rate == 0:
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)

        frame_count += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video (sampled at rate {sample_rate})")

    return frames


def detect_faces(frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
    """Detect faces in a frame using MediaPipe, falling back to Haar Cascades."""
    faces = []
    h, w, _ = frame.shape
    
    if mp_face_detection is not None:
        try:
            with mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=confidence_threshold
            ) as face_detection:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert normalized coordinates to pixel coordinates
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        
                        faces.append({
                            'bbox': (x, y, width, height),
                            'confidence': detection.score[0],
                            'face_region': frame[y:y+height, x:x+width]
                        })
                return faces
        except Exception as e:
            logger.warning(f"MediaPipe detection failed: {e}, falling back to Haar Cascades")
    
    # Fallback to Haar Cascades
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, width, height) in detected_faces:
        faces.append({
            'bbox': (x, y, width, height),
            'confidence': 0.9,  # Haar Cascades don't give confidence scores
            'face_region': frame[y:y+height, x:x+width]
        })
    
    return faces


def extract_face_regions(
    frames: List[np.ndarray],
    confidence_threshold: float = 0.5,
    max_faces: int = 1
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Pick the highest-confidence face from each frame and resize to 224×224."""
    face_regions = []
    face_metadata = []
    
    for frame_idx, frame in enumerate(frames):
        detections = detect_faces(frame, confidence_threshold)
        
        if detections:
            # Take the face with highest confidence (usually best quality)
            best_face = max(detections, key=lambda x: x['confidence'])
            
            face_region = best_face['face_region']
            
            if face_region.shape[0] > 20 and face_region.shape[1] > 20:
                face_resized = cv2.resize(face_region, (224, 224))
                face_regions.append(face_resized)
                
                face_metadata.append({
                    'frame_idx': frame_idx,
                    'bbox': best_face['bbox'],
                    'confidence': best_face['confidence'],
                    'original_size': face_region.shape
                })
    
    logger.info(f"Extracted {len(face_regions)} faces from {len(frames)} frames")
    
    return face_regions, face_metadata


def preprocess_faces(
    face_regions: List[np.ndarray],
    normalize: bool = True
) -> np.ndarray:
    """Convert face crops to float32 RGB arrays, optionally normalised to [0, 1]."""
    if not face_regions:
        return np.array([])
    
    processed = []
    
    for face in face_regions:
        if len(face.shape) == 2:  # grayscale → BGR
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if normalize:
            face_rgb = face_rgb.astype(np.float32) / 255.0
        else:
            face_rgb = face_rgb.astype(np.float32)
        
        processed.append(face_rgb)
    
    return np.array(processed)


def aggregate_predictions(
    predictions: List[float],
    method: str = 'mean'
) -> Tuple[float, Dict]:
    """Combine per-frame fake probabilities into a single video-level score."""
    if not predictions:
        return 0.5, {'method': method, 'num_frames': 0}
    
    predictions = np.array(predictions)
    
    if method == 'mean':
        score = float(np.mean(predictions))
    elif method == 'max':
        score = float(np.max(predictions))
    elif method == 'median':
        score = float(np.median(predictions))
    elif method == 'weighted':
        # Weight later frames more heavily (assumes forgeries have artifacts that become apparent)
        weights = np.linspace(1.0, 2.0, len(predictions))
        score = float(np.average(predictions, weights=weights))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    score = max(0.0, min(1.0, score))
    
    return score, {
        'method': method,
        'num_frames': len(predictions),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions))
    }


def load_celeb_df_video(video_path: str, label: str = None) -> Dict:
    """
    Load a Celeb-DF video with its metadata.
    
    Args:
        video_path: Path to video file
        label: 'real' or 'fake'
    
    Returns:
        Dictionary with video data and metadata
    """
    frames = extract_frames(video_path, sample_rate=5)  # Sample every 5th frame
    
    if not frames:
        logger.warning(f"No frames extracted from {video_path}")
        return None
    
    face_regions, face_metadata = extract_face_regions(frames)
    
    if not face_regions:
        logger.warning(f"No faces detected in {video_path}")
        return None
    
    processed_faces = preprocess_faces(face_regions, normalize=True)
    
    return {
        'video_path': video_path,
        'label': label,
        'frames': frames,
        'face_regions': face_regions,
        'processed_faces': processed_faces,
        'face_metadata': face_metadata,
        'num_frames': len(frames),
        'num_faces': len(face_regions)
    }
