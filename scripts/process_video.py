#!/usr/bin/env python3
"""Process classroom video with YOLO detection and ByteTrack tracking.

Usage:
    python scripts/process_video.py --source data/videos/classroom.mp4
    python scripts/process_video.py --source data/videos/classroom.mp4 --conf 0.4 --frame-skip 2 --show
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video.video_processor import VideoProcessor
from config import get_config


def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(
        description='ClassSight AI - Classroom Video Processing with Person Detection and Tracking'
    )
    parser.add_argument('--source', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default=None, help='Path for output video')
    parser.add_argument('--model', type=str, default=config.MODEL_NAME, help='YOLO model name')
    parser.add_argument('--conf', type=float, default=config.CONFIDENCE_THRESHOLD, help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=config.IOU_THRESHOLD, help='IoU threshold (0-1)')
    parser.add_argument('--frame-skip', type=int, default=config.FRAME_SKIP, help='Process every Nth frame')
    parser.add_argument('--show', action='store_true', default=config.SHOW_DISPLAY, help='Display video during processing')
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        processor = VideoProcessor(
            model_name=args.model,
            confidence=args.conf,
            iou=args.iou,
            frame_skip=args.frame_skip,
            show=args.show
        )
        
        report = processor.process(
            source=args.source,
            output_path=args.output
        )
        
        print("[INFO] Video processing completed successfully.")
        
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"[ERROR] I/O error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Processing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
