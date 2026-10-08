#!/usr/bin/env python3
"""Phase 3: Process classroom video with behavior analysis.

Usage:
    python scripts/process_video_behavior.py --source data/videos/classroom.mp4
    python scripts/process_video_behavior.py --source data/videos/classroom.mp4 --show
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.video.behavior_video_processor import BehaviorVideoProcessor
from config import get_config


def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(
        description='ClassSight AI Phase 3 - Video Processing with Behavior Analysis'
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
    parser.add_argument('--recognition-threshold', type=float,
                        default=config.RECOGNITION_THRESHOLD)
    parser.add_argument('--recognition-interval', type=int,
                        default=config.RECOGNITION_INTERVAL)
    # Phase 3 args
    parser.add_argument('--pose-model', type=str,
                        default=config.POSE_MODEL_NAME)
    parser.add_argument('--phone-threshold', type=float,
                        default=config.PHONE_CONFIDENCE)
    parser.add_argument('--phone-confirmation-frames', type=int,
                        default=config.PHONE_CONFIRMATION_FRAMES)
    parser.add_argument('--hand-confirmation-frames', type=int,
                        default=config.HAND_RAISE_CONFIRMATION_FRAMES)
    parser.add_argument('--head-pose-threshold', type=int,
                        default=config.HEAD_DOWN_CONFIRMATION_FRAMES,
                        help='Confirmation frames for head-down events')
    parser.add_argument('--event-confirmation-frames', type=int,
                        default=config.PHONE_CONFIRMATION_FRAMES,
                        help='Default confirmation frames for events')
    parser.add_argument('--min-event-duration', type=float,
                        default=config.MIN_EVENT_DURATION)
    parser.add_argument('--sleep-seconds', type=float,
                        default=config.SLEEP_CONFIRMATION_SECONDS,
                        help='Seconds of head-down for sleep indicator')
    parser.add_argument('--behavior-interval', type=int,
                        default=config.BEHAVIOR_INTERVAL,
                        help='Run behavior analysis every N frames')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    
    try:
        processor = BehaviorVideoProcessor(
            # Phase 1
            model_name=args.model,
            confidence=args.conf,
            iou=args.iou,
            frame_skip=args.frame_skip,
            show=args.show,
            # Phase 2
            insightface_model=config.INSIGHTFACE_MODEL,
            identities_file=str(config.IDENTITIES_FILE),
            embeddings_dir=str(config.EMBEDDINGS_DIR),
            recognition_threshold=args.recognition_threshold,
            confirmation_frames=config.CONFIRMATION_FRAMES,
            recognition_interval=args.recognition_interval,
            min_face_size=config.FACE_MIN_SIZE,
            retention_frames=config.IDENTITY_RETENTION_FRAMES,
            # Phase 3
            pose_model=args.pose_model,
            phone_confidence=args.phone_threshold,
            phone_class_id=config.PHONE_CLASS_ID,
            phone_proximity=config.PHONE_PROXIMITY_PIXELS,
            phone_confirmation_frames=args.phone_confirmation_frames,
            hand_confirmation_frames=args.hand_confirmation_frames,
            head_down_confirmation_frames=args.head_pose_threshold,
            sleep_confirmation_seconds=args.sleep_seconds,
            min_event_duration=args.min_event_duration,
            behavior_interval=args.behavior_interval
        )
        
        report = processor.process(
            source=args.source,
            output_path=args.output
        )
        
        print("[INFO] Phase 3 behavior analysis completed successfully.")
        
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
