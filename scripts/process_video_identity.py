#!/usr/bin/env python3
"""Phase 2: Process classroom video with person detection, tracking, and face recognition.

Usage:
    python scripts/process_video_identity.py --source data/videos/classroom.mp4
    python scripts/process_video_identity.py --source data/videos/classroom.mp4 --recognition-threshold 0.5 --confirmation-frames 5
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video.identity_video_processor import IdentityVideoProcessor
from config import get_config


def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(
        description='ClassSight AI Phase 2 - Video Processing with Face Recognition'
    )
    # Phase 1 args
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default=config.MODEL_NAME)
    parser.add_argument('--conf', type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument('--iou', type=float, default=config.IOU_THRESHOLD)
    parser.add_argument('--frame-skip', type=int, default=config.FRAME_SKIP)
    parser.add_argument('--show', action='store_true', default=config.SHOW_DISPLAY)
    # Phase 2 args
    parser.add_argument('--recognition-threshold', type=float, default=config.RECOGNITION_THRESHOLD,
                        help='Cosine similarity threshold for face recognition')
    parser.add_argument('--confirmation-frames', type=int, default=config.CONFIRMATION_FRAMES,
                        help='Consecutive matches needed to confirm identity')
    parser.add_argument('--recognition-interval', type=int, default=config.RECOGNITION_INTERVAL,
                        help='Run recognition every N processed frames')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    
    try:
        processor = IdentityVideoProcessor(
            model_name=args.model,
            confidence=args.conf,
            iou=args.iou,
            frame_skip=args.frame_skip,
            show=args.show,
            insightface_model=config.INSIGHTFACE_MODEL,
            identities_file=str(config.IDENTITIES_FILE),
            embeddings_dir=str(config.EMBEDDINGS_DIR),
            recognition_threshold=args.recognition_threshold,
            confirmation_frames=args.confirmation_frames,
            recognition_interval=args.recognition_interval,
            min_face_size=config.FACE_MIN_SIZE,
            retention_frames=config.IDENTITY_RETENTION_FRAMES
        )
        
        report = processor.process(
            source=args.source,
            output_path=args.output
        )
        
        print("[INFO] Phase 2 video processing completed successfully.")
        
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
