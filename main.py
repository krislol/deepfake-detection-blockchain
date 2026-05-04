"""
Main script for deepfake detection pipeline.
Handles video processing, model inference, and result aggregation.
"""

import sys
import os
from pathlib import Path
import json
import logging
import argparse
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.processing import (
    extract_frames,
    extract_face_regions,
    preprocess_faces,
    aggregate_predictions
)
from ml.model import DeepfakeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepfakeDetectionPipeline:
    """
    Complete deepfake detection pipeline.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        sample_rate: int = 5,
        confidence_threshold: float = 0.5
    ):
        self.detector = DeepfakeDetector(model_path=model_path, device=device)
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        logger.info("Pipeline initialized")
    
    def process_video(
        self,
        video_path: str,
        aggregation_method: str = 'mean',
        batch_size: int = 32
    ) -> Dict:
        """Extract frames, detect faces, run inference, and return a video-level verdict."""
        logger.info(f"Processing video: {video_path}")
        
        frames = extract_frames(video_path, sample_rate=self.sample_rate)
        if not frames:
            return {'success': False, 'error': 'Failed to extract frames from video', 'video_path': video_path}

        face_regions, face_metadata = extract_face_regions(frames, confidence_threshold=self.confidence_threshold)
        if not face_regions:
            return {'success': False, 'error': 'No faces detected in video', 'video_path': video_path, 'frames_processed': len(frames)}

        processed_faces = preprocess_faces(face_regions, normalize=True)
        frame_predictions, frame_metadata = self.detector.predict_batch(processed_faces, batch_size=batch_size)
        video_prediction, aggregation_stats = aggregate_predictions(frame_predictions, method=aggregation_method)

        result = {
            'success': True,
            'video_path': video_path,
            'video_prediction': {
                'fake_probability': float(video_prediction),
                'real_probability': float(1.0 - video_prediction),
                'classification': 'FAKE' if video_prediction > 0.5 else 'REAL',
                'confidence': float(max(video_prediction, 1.0 - video_prediction))
            },
            'aggregation_stats': aggregation_stats,
            'frame_predictions': frame_predictions,
            'num_frames_analyzed': len(frames),
            'num_faces_detected': len(face_regions),
            'frames': {
                'total_extracted': len(frames),
                'sample_rate': self.sample_rate
            }
        }
        
        logger.info(
            f"Video analysis complete. "
            f"Result: {result['video_prediction']['classification']} "
            f"(Probability: {video_prediction:.4f})"
        )
        
        return result
    
    def process_directory(
        self,
        directory_path: str,
        output_file: str = 'results.json',
        aggregation_method: str = 'mean'
    ) -> Dict:
        """Run detection on all videos in a directory and write results to JSON."""
        logger.info(f"Processing directory: {directory_path}")
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(directory_path).glob(f'*{ext}'))
        
        logger.info(f"Found {len(video_files)} video files")
        
        results = {
            'directory': directory_path,
            'total_videos': len(video_files),
            'videos': []
        }
        
        for idx, video_path in enumerate(video_files, 1):
            logger.info(f"Processing video {idx}/{len(video_files)}: {video_path.name}")
            
            result = self.process_video(
                str(video_path),
                aggregation_method=aggregation_method
            )
            results['videos'].append(result)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detection Pipeline'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file to analyze'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to directory containing videos'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=5,
        help='Sample every nth frame'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--aggregation',
        type=str,
        choices=['mean', 'max', 'median', 'weighted'],
        default='mean',
        help='Aggregation method for frame predictions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    pipeline = DeepfakeDetectionPipeline(
        model_path=args.model,
        device=args.device,
        sample_rate=args.sample_rate
    )

    if args.video:
        result = pipeline.process_video(
            args.video,
            aggregation_method=args.aggregation,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print("DEEPFAKE DETECTION RESULT")
        print("="*60)
        print(f"Video: {args.video}")
        
        if result['success']:
            pred = result['video_prediction']
            print(f"Classification: {pred['classification']}")
            print(f"Fake Probability: {pred['fake_probability']:.4f}")
            print(f"Real Probability: {pred['real_probability']:.4f}")
            print(f"Confidence: {pred['confidence']:.4f}")
            print(f"Frames Analyzed: {result['num_frames_analyzed']}")
            print(f"Faces Detected: {result['num_faces_detected']}")
        else:
            print(f"Error: {result['error']}")
        print("="*60 + "\n")
        
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")
        
    elif args.directory:
        results = pipeline.process_directory(
            args.directory,
            output_file=args.output,
            aggregation_method=args.aggregation
        )
        
        print("\n" + "="*60)
        print("DIRECTORY PROCESSING COMPLETE")
        print("="*60)
        print(f"Total Videos: {results['total_videos']}")
        
        successful = sum(1 for v in results['videos'] if v['success'])
        print(f"Successfully Processed: {successful}")
        print(f"Results saved to {args.output}")
        print("="*60 + "\n")
    
    else:
        logger.info("No video or directory specified. Running in demo mode...")
        test_videos = list(Path('data').glob('*.mp4')) if Path('data').exists() else []
        
        if test_videos:
            logger.info(f"Found test video: {test_videos[0]}")
            result = pipeline.process_video(
                str(test_videos[0]),
                aggregation_method=args.aggregation
            )
            
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Demo results saved to {args.output}")
        else:
            logger.warning(
                "No video or directory specified and no test videos found. "
                "Usage: python main.py --video path/to/video.mp4"
            )


if __name__ == '__main__':
    main()
