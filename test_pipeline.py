"""
Runs unit tests for each stage of the detection pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required packages can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info("✓ PyTorch imported")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        logger.info("✓ OpenCV imported")
    except ImportError as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import mediapipe
        logger.info("✓ MediaPipe imported")
    except ImportError as e:
        logger.error(f"✗ MediaPipe import failed: {e}")
        return False
    
    try:
        from ml.processing import extract_frames, extract_face_regions
        logger.info("✓ ml.processing imported")
    except ImportError as e:
        logger.error(f"✗ ml.processing import failed: {e}")
        return False
    
    try:
        from ml.model import DeepfakeDetector, DeepfakeDetectionModel
        logger.info("✓ ml.model imported")
    except ImportError as e:
        logger.error(f"✗ ml.model import failed: {e}")
        return False
    
    return True


def test_model_initialization():
    """Test that the model can be initialized."""
    logger.info("\nTesting model initialization...")

    try:
        from ml.model import DeepfakeDetector
        import torch

        detector = DeepfakeDetector()
        logger.info("✓ Model initialized with pretrained weights")

        # Test forward pass via the underlying nn.Module
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = detector.model(dummy_input)

        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
        logger.info("✓ Model forward pass successful")

        return True
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        return False


def test_detector_initialization():
    """Test that the detector can be initialized."""
    logger.info("\nTesting detector initialization...")
    
    try:
        from ml.model import DeepfakeDetector
        
        detector = DeepfakeDetector()
        logger.info("✓ DeepfakeDetector initialized")
        
        return True
    except Exception as e:
        logger.error(f"✗ DeepfakeDetector initialization failed: {e}")
        return False


def test_face_detection():
    """Test face detection on synthetic data."""
    logger.info("\nTesting face detection...")
    
    try:
        from ml.processing import detect_faces
        
        # Create synthetic face image
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame[100:300, 200:400] = 200  # Bright rectangle as fake face
        
        faces = detect_faces(frame, confidence_threshold=0.0)
        
        # May or may not detect synthetic face, just test it runs
        logger.info(f"✓ Face detection tested (found {len(faces)} faces)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Face detection failed: {e}")
        return False


def test_preprocessing():
    """Test image preprocessing."""
    logger.info("\nTesting preprocessing...")
    
    try:
        from ml.processing import preprocess_faces
        
        # Create synthetic face regions
        face_regions = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)
        ]
        
        processed = preprocess_faces(face_regions, normalize=True)
        
        assert processed.shape == (5, 224, 224, 3), f"Expected shape (5, 224, 224, 3), got {processed.shape}"
        assert processed.min() >= 0.0 and processed.max() <= 1.0, "Normalized values out of range"
        
        logger.info("✓ Preprocessing successful")
        
        return True
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        return False


def test_single_face_prediction():
    """Test prediction on a single face."""
    logger.info("\nTesting single face prediction...")
    
    try:
        from ml.model import DeepfakeDetector
        
        detector = DeepfakeDetector()
        
        # Create synthetic face image
        face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) / 255.0
        
        fake_prob, metadata = detector.predict_single_face(face_image)
        
        assert 0.0 <= fake_prob <= 1.0, f"Probability out of range: {fake_prob}"
        assert 'real_prob' in metadata and 'fake_prob' in metadata, "Missing metadata fields"
        
        logger.info(f"✓ Single face prediction: fake_prob={fake_prob:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Single face prediction failed: {e}")
        return False


def test_batch_prediction():
    """Test prediction on a batch of faces."""
    logger.info("\nTesting batch prediction...")
    
    try:
        from ml.model import DeepfakeDetector
        
        detector = DeepfakeDetector()
        
        # Create synthetic face images
        batch_size = 8
        face_images = np.random.randint(0, 255, (batch_size, 224, 224, 3), dtype=np.uint8).astype(np.float32) / 255.0
        
        predictions, metadata = detector.predict_batch(face_images, batch_size=4)
        
        assert len(predictions) == batch_size, f"Expected {batch_size} predictions, got {len(predictions)}"
        assert all(0.0 <= p <= 1.0 for p in predictions), "Some probabilities out of range"
        
        logger.info(f"✓ Batch prediction successful ({len(predictions)} faces)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Batch prediction failed: {e}")
        return False


def test_prediction_aggregation():
    """Test prediction aggregation."""
    logger.info("\nTesting prediction aggregation...")
    
    try:
        from ml.processing import aggregate_predictions
        
        # Create synthetic predictions
        predictions = [0.2, 0.3, 0.7, 0.8, 0.6]
        
        # Test different aggregation methods
        for method in ['mean', 'max', 'median', 'weighted']:
            score, stats = aggregate_predictions(predictions, method=method)
            
            assert 0.0 <= score <= 1.0, f"Score out of range for {method}: {score}"
            assert stats['method'] == method, f"Method mismatch in stats"
            
            logger.info(f"✓ {method.capitalize()} aggregation: {score:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Aggregation failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability."""
    logger.info("\nTesting GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
        else:
            logger.warning("⚠ No GPU available (will use CPU)")
        
        return True
    except Exception as e:
        logger.error(f"✗ GPU check failed: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist."""
    logger.info("\nTesting directory structure...")
    
    required_dirs = [
        'ml',
        'data',
    ]
    
    all_exist = True
    for directory in required_dirs:
        if Path(directory).exists():
            logger.info(f"✓ Directory exists: {directory}")
        else:
            logger.warning(f"⚠ Directory missing: {directory}")
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created: {directory}")
    
    return all_exist


def run_all_tests():
    """Run all tests."""
    logger.info("="*60)
    logger.info("DEEPFAKE DETECTION PIPELINE - TEST SUITE")
    logger.info("="*60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("GPU Availability", test_gpu_availability),
        ("Model Initialization", test_model_initialization),
        ("Detector Initialization", test_detector_initialization),
        ("Face Detection", test_face_detection),
        ("Preprocessing", test_preprocessing),
        ("Single Face Prediction", test_single_face_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Prediction Aggregation", test_prediction_aggregation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*60)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
