"""
Centralized configuration for deepfake detection system.
Modify these settings to customize the pipeline behavior.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = DATA_DIR / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODELS_DIR / 'deepfake_detector.pth'

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch

USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'

if USE_GPU:
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
else:
    GPU_NAME = None
    GPU_MEMORY = None

# ============================================================================
# VIDEO PROCESSING CONFIGURATION
# ============================================================================

# Frame extraction
FRAME_SAMPLE_RATE = 5              # Extract every nth frame (1=all, 5=every 5th)
FRAME_RESIZE_WIDTH = 640           # Resize frame width
FRAME_RESIZE_HEIGHT = 480          # Resize frame height
FRAME_RESIZE_RESOLUTION = (FRAME_RESIZE_WIDTH, FRAME_RESIZE_HEIGHT)

# Video formats to process
SUPPORTED_VIDEO_FORMATS = {
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'
}

# ============================================================================
# FACE DETECTION CONFIGURATION
# ============================================================================

# MediaPipe Face Detection
FACE_DETECTION_MODEL_SELECTION = 0  # 0=short range (0-2m), 1=full range (0-5m)
FACE_DETECTION_CONFIDENCE = 0.5     # Minimum confidence threshold
FACE_MIN_WIDTH = 20                 # Minimum face region width (pixels)
FACE_MIN_HEIGHT = 20                # Minimum face region height (pixels)
FACE_STANDARD_SIZE = 224            # Standard size for model input

# Maximum faces to process per video (limit for efficiency)
MAX_FACES_PER_VIDEO = 1000

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model architecture
MODEL_ARCHITECTURE = 'EfficientNetB0'  # EfficientNet-B0 backbone
NUM_CLASSES = 2                        # Binary classification: Real/Fake
PRETRAINED = True                      # Use ImageNet pretrained weights

# Dropout rates
DROPOUT_RATE = 0.2
DROPOUT_CLASSIFIER = 0.2

# Model input size
MODEL_INPUT_SIZE = 224

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Batch processing
DEFAULT_BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 32

# Prediction threshold for classification
FAKE_PROBABILITY_THRESHOLD = 0.5  # Predictions >= 0.5 are classified as FAKE

# Aggregation method for frame predictions
DEFAULT_AGGREGATION_METHOD = 'mean'  # Options: mean, max, median, weighted

# Normalization values (ImageNet)
IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
EPSILON = 1e-8

# Training schedule
EPOCHS = 10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 32

# Learning rate scheduler
LR_SCHEDULER = 'ReduceLROnPlateau'  # Options: ReduceLROnPlateau, StepLR, CosineAnnealingLR
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

# Early stopping
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 5

# Data augmentation
AUGMENTATION_ROTATION = 10
AUGMENTATION_BRIGHTNESS = 0.2
AUGMENTATION_CONTRAST = 0.2
AUGMENTATION_SATURATION = 0.2
AUGMENTATION_HFLIP_PROB = 0.5

# Train/Val split
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val

# ============================================================================
# CELEB-DF DATASET CONFIGURATION
# ============================================================================

CELEB_DF_PATH = Path.home() / 'Celeb-DF-v2'
CELEB_DF_REAL_DIR = CELEB_DF_PATH / 'YouTube-real'
CELEB_DF_FAKE_DIR = CELEB_DF_PATH / 'synthesized_videos'
CELEB_DF_FACE_CACHE = CELEB_DF_PATH / 'face_cache'

# Maximum faces to extract per video (for memory efficiency)
CELEB_DF_FACES_PER_VIDEO = 10

# ============================================================================
# FASTAPI CONFIGURATION
# ============================================================================

# API server
API_HOST = '0.0.0.0'
API_PORT = 8000
API_RELOAD = True
API_LOG_LEVEL = 'info'

# File upload
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
TEMP_UPLOAD_DIR = PROJECT_ROOT / 'temp_uploads'

# Response configuration
INCLUDE_FRAME_PREDICTIONS = True  # Include per-frame predictions in API response
INCLUDE_TIMING_INFO = True        # Include processing time information

# ============================================================================
# BLOCKCHAIN CONFIGURATION
# ============================================================================

# Polygon Network
POLYGON_RPC_URL = os.getenv(
    'POLYGON_RPC_URL',
    'https://polygon-rpc.com'
)
POLYGON_CHAIN_ID = 137
POLYGON_NETWORK = 'polygon-mainnet'

# Contract configuration
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
CONTRACT_GAS_LIMIT = 300000
CONTRACT_GAS_PRICE = 'auto'  # or specific value in Wei

# File hashing
HASH_ALGORITHM = 'sha256'

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / 'deepfake_detection.log'
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ============================================================================
# RESULT OUTPUT CONFIGURATION
# ============================================================================

# Include detailed statistics in results
INCLUDE_AGGREGATION_STATS = True
INCLUDE_FRAME_METADATA = False  # Set to False for smaller output files
INCLUDE_PROCESSING_TIME = True

# Result precision (decimal places)
RESULT_PRECISION = 4

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

# Number of worker threads for data loading
NUM_WORKERS = 4

# Seed for reproducibility
RANDOM_SEED = 42

# Cache face detections (speeds up re-processing)
CACHE_FACE_DETECTIONS = True
FACE_CACHE_DIR = DATA_DIR / 'face_cache'

# Performance monitoring
ENABLE_TIMING = True
ENABLE_MEMORY_MONITORING = False

# Debug mode (verbose output)
DEBUG_MODE = False

# ============================================================================
# MODEL EVALUATION CONFIGURATION
# ============================================================================

# Evaluation metrics
COMPUTE_ROC_AUC = True
COMPUTE_PR_AUC = True
COMPUTE_CONFUSION_MATRIX = True

# Test set split
TEST_SET_SIZE = 0.2

# ============================================================================
# PRODUCTION CONFIGURATION
# ============================================================================

PRODUCTION = False

if PRODUCTION:
    API_RELOAD = False
    LOG_LEVEL = 'WARNING'
    DEBUG_MODE = False
    NUM_WORKERS = 8

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config_summary():
    """Get a summary of key configuration settings."""
    return {
        'device': DEVICE,
        'gpu_available': USE_GPU,
        'model_path': str(DEFAULT_MODEL_PATH),
        'frame_sample_rate': FRAME_SAMPLE_RATE,
        'batch_size': DEFAULT_BATCH_SIZE,
        'model': MODEL_ARCHITECTURE,
        'api_host': API_HOST,
        'api_port': API_PORT,
        'production': PRODUCTION
    }

def print_config():
    """Print current configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"{key:.<40} {value}")
    
    print("="*60 + "\n")

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if DEFAULT_BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    if FRAME_SAMPLE_RATE <= 0:
        errors.append("FRAME_SAMPLE_RATE must be positive")
    
    if not (0.0 <= FACE_DETECTION_CONFIDENCE <= 1.0):
        errors.append("FACE_DETECTION_CONFIDENCE must be between 0 and 1")
    
    if not (0.0 <= FAKE_PROBABILITY_THRESHOLD <= 1.0):
        errors.append("FAKE_PROBABILITY_THRESHOLD must be between 0 and 1")
    
    if not (0.0 < TRAIN_VAL_SPLIT < 1.0):
        errors.append("TRAIN_VAL_SPLIT must be between 0 and 1")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == '__main__':
    print_config()
    
    # Validate
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has errors")
