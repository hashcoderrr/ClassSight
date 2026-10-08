#!/usr/bin/env python3
"""Phase 4: Analyze engagement from behavior report or video.

Usage:
    # From existing Phase 3 behavior report
    python scripts/analyze_engagement.py --report outputs/reports/behavior_report_*.json

    # Full pipeline from video
    python scripts/analyze_engagement.py --source data/videos/classroom.mp4

    # Custom weights
    python scripts/analyze_engagement.py --report report.json --forward-weight 0.35
"""
import sys
import json
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config


def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(
        description='ClassSight AI Phase 4 - Engagement Analytics'
    )
    
    # Input mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--report', type=str,
        help='Path to Phase 3 behavior report JSON'
    )
    group.add_argument(
        '--source', type=str,
        help='Path to video for full Phase 1+2+3+4 processing'
    )
    
    # Phase 4 settings
    parser.add_argument(
        '--time-window', type=float,
        default=config.ENGAGEMENT_TIME_WINDOW,
        help='Time window for timeline analysis (seconds)'
    )
    parser.add_argument(
        '--min-observation', type=float,
        default=config.MIN_OBSERVATION_SECONDS,
        help='Minimum observation seconds for scoring'
    )
    parser.add_argument(
        '--forward-weight', type=float, default=None,
        help='Override weight for forward_ratio'
    )
    parser.add_argument(
        '--phone-weight', type=float, default=None,
        help='Override weight for phone_ratio'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating visualization plots'
    )
    
    # Video processing settings (only used with --source)
    parser.add_argument('--model', type=str, default=config.MODEL_NAME)
    parser.add_argument('--conf', type=float,
                        default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument('--show', action='store_true',
                        default=config.SHOW_DISPLAY)
    
    return parser.parse_args()


def analyze_from_report(args):
    """Run engagement analysis on existing behavior report."""
    from src.engagement.engagement_engine import EngagementEngine
    from src.engagement.engagement_metrics import EngagementReporter
    
    config = get_config()
    report_path = Path(args.report)
    
    if not report_path.exists():
        print(f"[ERROR] Report not found: {report_path}")
        sys.exit(1)
    
    with open(report_path) as f:
        behavior_report = json.load(f)
    
    # Extract events from behavior report
    events = []
    per_student = behavior_report.get('per_student_events', {})
    for key, event_list in per_student.items():
        events.extend(event_list)
    
    # Session duration from report
    total_frames = behavior_report.get('total_frames', 0)
    fps = behavior_report.get('average_fps', 25.0)
    # Try to compute duration from frames/fps in report
    # The behavior report doesn't store original video fps directly,
    # so we estimate from total_frames / processed average fps
    session_duration = total_frames / fps if fps > 0 else 0
    
    # Build weights
    weights = dict(config.ENGAGEMENT_WEIGHTS)
    if args.forward_weight is not None:
        weights['forward_ratio'] = args.forward_weight
    if args.phone_weight is not None:
        weights['phone_ratio'] = args.phone_weight
    
    # Run engagement analysis
    engine = EngagementEngine(
        weights=weights,
        base_score=config.ENGAGEMENT_BASE_SCORE,
        level_thresholds=config.ENGAGEMENT_LEVEL_THRESHOLDS,
        raised_hand_rate_cap=config.RAISED_HAND_RATE_CAP,
        time_window=args.time_window,
        min_observation_seconds=args.min_observation,
        min_data_coverage=config.MIN_DATA_COVERAGE,
    )
    
    result = engine.analyze(
        events=events,
        session_duration=session_duration,
    )
    
    # Generate report
    reporter = EngagementReporter()
    report_dir = report_path.parent
    report = reporter.generate_report(
        analysis_result=result,
        video_name=behavior_report.get('video', ''),
        resolution=behavior_report.get('resolution', ''),
        total_frames=total_frames,
        fps=fps,
        session_duration=session_duration,
        output_dir=report_dir,
        save_plots=not args.no_plots,
    )
    
    # Print summary
    classroom = result.get('classroom', {})
    print(f"\n{'='*60}")
    print(f"Phase 4 Engagement Analysis Complete")
    print(f"{'='*60}")
    print(f"Source report: {report_path.name}")
    print(f"Students scored: {classroom.get('scored_students', 0)}")
    print(f"Average engagement: {classroom.get('average_engagement', 0):.1f}")
    print(f"Median engagement: {classroom.get('median_engagement', 0):.1f}")
    dist = classroom.get('engagement_distribution', {})
    if dist:
        print(f"Distribution: HIGH={dist.get('HIGH',0)} "
              f"MODERATE={dist.get('MODERATE',0)} "
              f"LOW={dist.get('LOW',0)}")
    print(f"{'='*60}\n")
    
    return report


def analyze_from_video(args):
    """Run full Phase 1+2+3+4 pipeline on video."""
    from src.video.engagement_video_processor import EngagementVideoProcessor
    
    config = get_config()
    
    # Build weights
    weights = dict(config.ENGAGEMENT_WEIGHTS)
    if args.forward_weight is not None:
        weights['forward_ratio'] = args.forward_weight
    if args.phone_weight is not None:
        weights['phone_ratio'] = args.phone_weight
    
    processor = EngagementVideoProcessor(
        # Phase 1
        model_name=args.model,
        confidence=args.conf,
        iou=config.IOU_THRESHOLD,
        frame_skip=config.FRAME_SKIP,
        show=args.show,
        # Phase 2
        insightface_model=config.INSIGHTFACE_MODEL,
        identities_file=str(config.IDENTITIES_FILE),
        embeddings_dir=str(config.EMBEDDINGS_DIR),
        recognition_threshold=config.RECOGNITION_THRESHOLD,
        confirmation_frames=config.CONFIRMATION_FRAMES,
        recognition_interval=config.RECOGNITION_INTERVAL,
        min_face_size=config.FACE_MIN_SIZE,
        retention_frames=config.IDENTITY_RETENTION_FRAMES,
        # Phase 3
        pose_model=config.POSE_MODEL_NAME,
        phone_confidence=config.PHONE_CONFIDENCE,
        phone_class_id=config.PHONE_CLASS_ID,
        phone_proximity=config.PHONE_PROXIMITY_PIXELS,
        phone_confirmation_frames=config.PHONE_CONFIRMATION_FRAMES,
        hand_confirmation_frames=config.HAND_RAISE_CONFIRMATION_FRAMES,
        head_down_confirmation_frames=config.HEAD_DOWN_CONFIRMATION_FRAMES,
        sleep_confirmation_seconds=config.SLEEP_CONFIRMATION_SECONDS,
        min_event_duration=config.MIN_EVENT_DURATION,
        behavior_interval=config.BEHAVIOR_INTERVAL,
        # Phase 4
        engagement_weights=weights,
        engagement_base_score=config.ENGAGEMENT_BASE_SCORE,
        engagement_level_thresholds=config.ENGAGEMENT_LEVEL_THRESHOLDS,
        raised_hand_rate_cap=config.RAISED_HAND_RATE_CAP,
        engagement_time_window=args.time_window,
        min_observation_seconds=args.min_observation,
        min_data_coverage=config.MIN_DATA_COVERAGE,
    )
    
    report = processor.process(source=args.source)
    print("[INFO] Phase 4 engagement analysis completed successfully.")
    return report


def main():
    args = parse_args()
    
    try:
        if args.report:
            analyze_from_report(args)
        else:
            analyze_from_video(args)
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
