#!/usr/bin/env python3
"""Test YOLO person detection on a single image.

Usage:
    python scripts/test_image.py --source data/images/classroom.jpg
    python scripts/test_image.py --source data/images/classroom.jpg --conf 0.5 --show
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
from src.detection.yolo_detector import YOLODetector
from src.utils.helpers import (
    ensure_dir, validate_image_path, get_color, 
    draw_detection
)
from config import get_config


def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(description='ClassSight AI - Image Person Detection')
    parser.add_argument('--source', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default=config.MODEL_NAME, help='YOLO model name')
    parser.add_argument('--conf', type=float, default=config.CONFIDENCE_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=config.IOU_THRESHOLD, help='IoU threshold')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--show', action='store_true', help='Display result window')
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Validate input
        image_path = validate_image_path(args.source)
        print(f"\n[INFO] Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        print(f"[INFO] Image size: {w}x{h}")
        
        # Initialize detector
        detector = YOLODetector(
            model_name=args.model,
            confidence=args.conf,
            iou=args.iou
        )
        
        # Run detection
        result = detector.detect(image)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Detection Results")
        print(f"{'='*50}")
        print(f"Persons detected: {result['count']}")
        print(f"Inference time: {result['inference_time']*1000:.1f}ms")
        print()
        
        for i, (box, conf) in enumerate(zip(result['boxes'], result['confidences'])):
            print(f"  Person {i+1}: confidence={conf:.4f}, "
                  f"bbox=[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")
            
            # Draw on image
            color = get_color(i)
            label = f"Student ({conf:.2f})"
            draw_detection(image, box, label, color)
        
        print(f"{'='*50}\n")
        
        # Save output
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = ensure_dir(project_root / 'outputs' / 'images')
            output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
        
        ensure_dir(output_path.parent)
        cv2.imwrite(str(output_path), image)
        print(f"[INFO] Output saved: {output_path}")
        
        # Show if requested
        if args.show:
            cv2.imshow('ClassSight AI - Detection', image)
            print("[INFO] Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
