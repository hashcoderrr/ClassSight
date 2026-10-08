"""Phase 4 video processor with engagement analytics.

Inherits from BehaviorVideoProcessor (Phase 3) and adds:
1. Per-track frame counting for accurate visible duration
2. Post-processing engagement analysis
3. Engagement overlay on annotated video
4. Combined Phase 1+2+3+4 report generation

Does NOT modify any Phase 1, 2, or 3 modules — inherits and extends.
"""

import cv2
import time
from pathlib import Path
from typing import Optional

from src.video.behavior_video_processor import BehaviorVideoProcessor
from src.engagement.engagement_engine import EngagementEngine
from src.engagement.engagement_metrics import EngagementReporter
from src.utils.helpers import ensure_dir, validate_video_path, format_time


class EngagementVideoProcessor(BehaviorVideoProcessor):
    """Video processor with detection, tracking, identity, behavior, and engagement.
    
    Extends BehaviorVideoProcessor to add:
    - Per-track frame counter for accurate visibility tracking
    - Post-processing engagement analysis from behavior events
    - Engagement indicators in annotated output
    """
    
    def __init__(self,
                 # Phase 4 settings
                 engagement_weights: Optional[dict] = None,
                 engagement_base_score: float = 50.0,
                 engagement_level_thresholds: Optional[dict] = None,
                 raised_hand_rate_cap: float = 1.0,
                 engagement_time_window: float = 60.0,
                 min_observation_seconds: float = 30.0,
                 min_data_coverage: float = 0.30,
                 # All Phase 1/2/3 settings forwarded
                 **kwargs):
        """Initialize with Phase 4 engagement settings plus all Phase 1/2/3 params."""
        super().__init__(**kwargs)
        
        self._engagement_params = {
            'weights': engagement_weights,
            'base_score': engagement_base_score,
            'level_thresholds': engagement_level_thresholds,
            'raised_hand_rate_cap': raised_hand_rate_cap,
            'time_window': engagement_time_window,
            'min_observation_seconds': min_observation_seconds,
            'min_data_coverage': min_data_coverage,
        }
        
        # Per-track frame counter: {track_id: frame_count}
        self._track_frame_counts: dict[int, int] = {}
    
    def process(self, source: str, output_path: Optional[str] = None) -> dict:
        """Process video with full Phase 1+2+3+4 pipeline.
        
        Overrides parent to add per-track counting and post-processing
        engagement analysis.
        """
        source_path = validate_video_path(source)
        
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        
        metadata = self.get_video_metadata(cap)
        self._init_behavior_engine(metadata['fps'])
        
        # Print header
        print(f"\n{'='*60}")
        print(f"ClassSight AI - Phase 4: Engagement Analytics")
        print(f"{'='*60}")
        print(f"Source: {source_path.name}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"FPS: {metadata['fps']:.1f}")
        print(f"Frames: {metadata['frame_count']}")
        print(f"Duration: {format_time(metadata['duration'])}")
        print(f"Detection Model: {self.model_name}")
        print(f"Pose Model: {self._phase3_params['pose_model']}")
        print(f"Enrolled Students: {self.recognizer.enrolled_count}")
        print(f"Engagement Time Window: {self._engagement_params['time_window']}s")
        print(f"{'='*60}\n")
        
        # Setup output
        if output_path is None:
            output_dir = ensure_dir(
                Path(__file__).parent.parent.parent / 'outputs' / 'videos'
            )
            output_path = str(
                output_dir / f"{source_path.stem}_engagement.mp4"
            )
        else:
            ensure_dir(Path(output_path).parent)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, metadata['fps'],
            (metadata['width'], metadata['height'])
        )
        
        if not out.isOpened():
            cap.release()
            raise IOError(f"Cannot create output video: {output_path}")
        
        self.metrics.start()
        frame_number = 0
        processed_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                if frame_number % self.frame_skip != 0:
                    out.write(frame)
                    continue
                
                processed_count += 1
                frame_start = time.time()
                
                # Phase 1: Track persons
                track_result = self.tracker.track(frame)
                
                # Count frames per track
                for tid in track_result['track_ids']:
                    if tid != -1:
                        self._track_frame_counts[tid] = (
                            self._track_frame_counts.get(tid, 0) + 1
                        )
                
                # Phase 2: Face recognition (every N frames)
                run_recognition = (
                    processed_count % self.recognition_interval == 0
                    and self.recognizer.enrolled_count > 0
                )
                
                if run_recognition and track_result['count'] > 0:
                    recog_results = self.recognizer.recognize_in_frame(
                        frame,
                        track_result['boxes'],
                        track_result['track_ids']
                    )
                    for rr in recog_results:
                        self.identity_manager.update(
                            track_id=rr.track_id,
                            student_id=rr.student_id,
                            name=rr.name,
                            similarity=rr.similarity,
                            is_known=(rr.status == 'MATCHED'),
                            has_face=(
                                rr.status != 'NO_FACE'
                                and rr.status != 'SMALL_FACE'
                            )
                        )
                else:
                    for tid in track_result['track_ids']:
                        if tid != -1:
                            self.identity_manager.update_no_recognition(tid)
                
                # Phase 3: Behavior analysis
                behavior_result = {}
                run_behavior = (
                    processed_count % self.behavior_interval == 0
                    and track_result['count'] > 0
                )
                
                if run_behavior:
                    identity_mappings = (
                        self.identity_manager.get_all_mappings()
                    )
                    behavior_result = self.behavior_engine.process_frame(
                        frame, track_result, identity_mappings, frame_number
                    )
                    self.metrics.update_behavior(
                        behavior_result,
                        behavior_result.get('analysis_time', 0.0)
                    )
                
                frame_time = time.time() - frame_start
                
                self.metrics.update_frame(
                    frame_time=frame_time,
                    student_count=track_result['count'],
                    track_ids=track_result['track_ids']
                )
                
                # Draw annotations (reuse parent drawing)
                active_events = behavior_result.get('active_events', {})
                head_results = behavior_result.get('head_results', {})
                
                for box, conf, tid in zip(
                    track_result['boxes'],
                    track_result['confidences'],
                    track_result['track_ids']
                ):
                    identity = (
                        self.identity_manager.get_identity(tid)
                        if tid != -1 else None
                    )
                    student_name = None
                    if identity and identity.confirmed_id:
                        student_name = identity.confirmed_name
                    
                    tid_events = active_events.get(tid, {})
                    self._draw_behavior_box(
                        frame, box, tid, student_name,
                        tid_events, head_results.get(tid, {})
                    )
                
                # Draw overlay
                from src.utils.helpers import draw_info_overlay
                id_summary = self.identity_manager.get_summary()
                event_counts = (
                    self.behavior_engine.get_event_counts()
                    if self.behavior_engine else {}
                )
                info = {
                    'FPS': f"{self.metrics.current_fps:.1f}",
                    'Students': str(track_result['count']),
                    'Identified': (
                        f"{id_summary['confirmed']}/"
                        f"{id_summary.get('total_tracks', 0)}"
                    ),
                    'Frame': (
                        f"{frame_number}/{metadata['frame_count']}"
                    ),
                    'Events': str(
                        sum(event_counts.values())
                    ) if event_counts else '0',
                }
                draw_info_overlay(frame, info)
                
                out.write(frame)
                
                if self.show:
                    cv2.imshow('ClassSight AI - Engagement', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] User pressed Q - stopping.")
                        break
                
                if frame_number % 100 == 0:
                    print(
                        f"  Frame {frame_number}/{metadata['frame_count']} "
                        f"| FPS: {self.metrics.current_fps:.1f} "
                        f"| Students: {track_result['count']} "
                        f"| Events: "
                        f"{sum(event_counts.values()) if event_counts else 0}"
                    )
        
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt - stopping.")
        
        finally:
            self.metrics.stop()
            if cap.isOpened():
                cap.release()
            if out.isOpened():
                out.release()
            if self.show:
                cv2.destroyAllWindows()
        
        # Finalize Phase 3 events
        identity_mappings = self.identity_manager.get_all_mappings()
        all_events = (
            self.behavior_engine.finalize(identity_mappings)
            if self.behavior_engine else []
        )
        
        # Phase 4: Engagement analysis
        fps = metadata['fps']
        track_visibility = {
            tid: count / fps
            for tid, count in self._track_frame_counts.items()
        } if fps > 0 else {}
        
        engagement_engine = EngagementEngine(**self._engagement_params)
        engagement_result = engagement_engine.analyze(
            events=all_events,
            session_duration=metadata['duration'],
            identity_mappings=identity_mappings,
            track_visibility=track_visibility,
            identity_summary=self.identity_manager.get_summary(),
        )
        
        # Generate engagement report
        report_dir = ensure_dir(
            Path(__file__).parent.parent.parent / 'outputs' / 'reports'
        )
        reporter = EngagementReporter()
        report = reporter.generate_report(
            analysis_result=engagement_result,
            video_name=source_path.name,
            resolution=metadata['resolution'],
            total_frames=metadata['frame_count'],
            fps=fps,
            session_duration=metadata['duration'],
            output_dir=report_dir,
        )
        
        # Print summary
        classroom = engagement_result.get('classroom', {})
        students = engagement_result.get('students', {})
        
        print(f"\n{'='*60}")
        print(f"Phase 4 Engagement Analysis Complete")
        print(f"{'='*60}")
        print(f"Frames processed: {self.metrics.total_frames_processed}")
        print(f"Average FPS: {self.metrics.average_fps:.1f}")
        print(f"Unique tracks: {len(self.metrics.all_track_ids)}")
        print(f"Total behavior events: {len(all_events)}")
        print(f"Students scored: {classroom.get('scored_students', 0)}")
        print(f"Average engagement: {classroom.get('average_engagement', 0):.1f}")
        print(f"Median engagement: {classroom.get('median_engagement', 0):.1f}")
        dist = classroom.get('engagement_distribution', {})
        if dist:
            print(f"Distribution: HIGH={dist.get('HIGH',0)} "
                  f"MODERATE={dist.get('MODERATE',0)} "
                  f"LOW={dist.get('LOW',0)}")
        print(f"Processing time: {format_time(self.metrics.total_processing_time)}")
        print(f"Output video: {output_path}")
        print(f"{'='*60}\n")
        
        return report
