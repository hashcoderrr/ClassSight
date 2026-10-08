"""Phase 3 video processor with behavior analysis.

Composes Phase 1 (YOLO + ByteTrack tracking), Phase 2 (face recognition),
and Phase 3 (behavior analysis) into a single pipeline.

Does NOT modify any Phase 1 or Phase 2 modules — composes them.
"""

import cv2
import time
from pathlib import Path
from typing import Optional

from src.tracking.student_tracker import StudentTracker
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.identity_manager import IdentityManager
from src.behavior.behavior_engine import BehaviorEngine
from src.evaluation.behavior_metrics import BehaviorMetrics
from src.utils.helpers import (
    ensure_dir, validate_video_path, get_color,
    draw_detection, draw_info_overlay, format_time
)


# Behavior-specific colors (BGR)
BEHAVIOR_COLORS = {
    'phone_usage': (0, 0, 255),      # Red
    'raised_hand': (255, 255, 0),     # Cyan
    'head_down': (0, 165, 255),       # Orange
    'head_left': (255, 200, 100),     # Light blue
    'head_right': (255, 200, 100),    # Light blue
    'possible_sleeping': (0, 0, 200), # Dark red
    'head_forward': (0, 255, 0),      # Green
}


class BehaviorVideoProcessor:
    """Video processor with detection, tracking, identity, and behavior analysis.
    
    Pipeline per frame:
    1. YOLO + ByteTrack → person boxes + track IDs (Phase 1)
    2. Every N frames: Face recognition → identity mapping (Phase 2)
    3. Behavior analysis: pose, phone, hand, head, sleep (Phase 3)
    4. Temporal event confirmation
    5. Draw annotated overlays
    6. Write output video + generate behavior report
    """
    
    def __init__(self,
                 # Phase 1
                 model_name: str = 'yolo11n.pt',
                 confidence: float = 0.4,
                 iou: float = 0.5,
                 frame_skip: int = 1,
                 show: bool = False,
                 # Phase 2
                 insightface_model: str = 'buffalo_l',
                 identities_file: str = 'data/identities/students.json',
                 embeddings_dir: str = 'data/identities/embeddings',
                 recognition_threshold: float = 0.4,
                 confirmation_frames: int = 3,
                 recognition_interval: int = 5,
                 min_face_size: int = 40,
                 retention_frames: int = 30,
                 # Phase 3
                 pose_model: str = 'yolo11n-pose.pt',
                 phone_confidence: float = 0.3,
                 phone_class_id: int = 67,
                 phone_proximity: int = 100,
                 phone_confirmation_frames: int = 3,
                 hand_confirmation_frames: int = 2,
                 head_down_confirmation_frames: int = 5,
                 sleep_confirmation_seconds: float = 5.0,
                 min_event_duration: float = 0.5,
                 behavior_interval: int = 1):
        
        self.model_name = model_name
        self.frame_skip = max(1, frame_skip)
        self.show = show
        self.recognition_interval = max(1, recognition_interval)
        self.behavior_interval = max(1, behavior_interval)
        
        # Phase 1
        self.tracker = StudentTracker(
            model_name=model_name,
            confidence=confidence,
            iou=iou
        )
        
        # Phase 2
        self.recognizer = FaceRecognizer(
            model_name=insightface_model,
            identities_file=identities_file,
            embeddings_dir=embeddings_dir,
            threshold=recognition_threshold,
            min_face_size=min_face_size
        )
        self.identity_manager = IdentityManager(
            confirmation_frames=confirmation_frames,
            retention_frames=retention_frames
        )
        
        # Phase 3 (fps set after video metadata is read)
        self._phase3_params = {
            'pose_model': pose_model,
            'phone_model': model_name,
            'phone_confidence': phone_confidence,
            'phone_class_id': phone_class_id,
            'phone_proximity': phone_proximity,
            'phone_confirmation_frames': phone_confirmation_frames,
            'hand_confirmation_frames': hand_confirmation_frames,
            'head_down_confirmation_frames': head_down_confirmation_frames,
            'sleep_confirmation_seconds': sleep_confirmation_seconds,
            'min_event_duration': min_event_duration,
        }
        self.behavior_engine = None  # Initialized once FPS is known
        self.metrics = BehaviorMetrics()
    
    def _init_behavior_engine(self, fps: float):
        """Initialize behavior engine once video FPS is known."""
        self.behavior_engine = BehaviorEngine(
            fps=fps,
            **self._phase3_params
        )
    
    def get_video_metadata(self, cap: cv2.VideoCapture) -> dict:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        return {
            'width': width, 'height': height, 'fps': fps,
            'frame_count': frame_count, 'duration': duration,
            'resolution': f"{width}x{height}"
        }
    
    def process(self, source: str, output_path: Optional[str] = None) -> dict:
        """Process a video with detection, tracking, identity, and behavior analysis."""
        source_path = validate_video_path(source)
        
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        
        metadata = self.get_video_metadata(cap)
        self._init_behavior_engine(metadata['fps'])
        
        # Print header
        print(f"\n{'='*60}")
        print(f"ClassSight AI - Phase 3: Behavior Analysis")
        print(f"{'='*60}")
        print(f"Source: {source_path.name}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"FPS: {metadata['fps']:.1f}")
        print(f"Frames: {metadata['frame_count']}")
        print(f"Duration: {format_time(metadata['duration'])}")
        print(f"Detection Model: {self.model_name}")
        print(f"Pose Model: {self._phase3_params['pose_model']}")
        print(f"Enrolled Students: {self.recognizer.enrolled_count}")
        print(f"Behavior Interval: every {self.behavior_interval} frames")
        print(f"{'='*60}\n")
        
        # Setup output
        if output_path is None:
            output_dir = ensure_dir(Path(__file__).parent.parent.parent / 'outputs' / 'videos')
            output_path = str(output_dir / f"{source_path.stem}_behavior.mp4")
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
                
                # Phase 2: Face recognition (every N frames)
                run_recognition = (processed_count % self.recognition_interval == 0
                                    and self.recognizer.enrolled_count > 0)
                
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
                            has_face=(rr.status != 'NO_FACE' and rr.status != 'SMALL_FACE')
                        )
                else:
                    for tid in track_result['track_ids']:
                        if tid != -1:
                            self.identity_manager.update_no_recognition(tid)
                
                # Phase 3: Behavior analysis
                behavior_result = {}
                run_behavior = (processed_count % self.behavior_interval == 0
                                 and track_result['count'] > 0)
                
                if run_behavior:
                    identity_mappings = self.identity_manager.get_all_mappings()
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
                
                # Draw annotations
                active_events = behavior_result.get('active_events', {})
                head_results = behavior_result.get('head_results', {})
                
                for box, conf, tid in zip(
                    track_result['boxes'],
                    track_result['confidences'],
                    track_result['track_ids']
                ):
                    identity = self.identity_manager.get_identity(tid) if tid != -1 else None
                    student_name = None
                    if identity and identity.confirmed_id:
                        student_name = identity.confirmed_name
                    
                    tid_events = active_events.get(tid, {})
                    
                    self._draw_behavior_box(
                        frame, box, tid, student_name,
                        tid_events, head_results.get(tid, {})
                    )
                
                # Draw overlay
                id_summary = self.identity_manager.get_summary()
                event_counts = self.behavior_engine.get_event_counts() if self.behavior_engine else {}
                info = {
                    'FPS': f"{self.metrics.current_fps:.1f}",
                    'Students': str(track_result['count']),
                    'Identified': f"{id_summary['confirmed']}/{id_summary.get('total_tracks', 0)}",
                    'Frame': f"{frame_number}/{metadata['frame_count']}",
                    'Events': str(sum(event_counts.values())) if event_counts else '0'
                }
                draw_info_overlay(frame, info)
                
                out.write(frame)
                
                if self.show:
                    cv2.imshow('ClassSight AI - Behavior', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] User pressed Q - stopping.")
                        break
                
                if frame_number % 100 == 0:
                    print(f"  Frame {frame_number}/{metadata['frame_count']} "
                          f"| FPS: {self.metrics.current_fps:.1f} "
                          f"| Students: {track_result['count']} "
                          f"| Events: {sum(event_counts.values()) if event_counts else 0}")
        
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
        
        # Finalize events
        identity_mappings = self.identity_manager.get_all_mappings()
        all_events = self.behavior_engine.finalize(identity_mappings) if self.behavior_engine else []
        event_counts = self.behavior_engine.get_event_counts() if self.behavior_engine else {}
        
        # Generate report
        report_dir = ensure_dir(Path(__file__).parent.parent.parent / 'outputs' / 'reports')
        report = self.metrics.generate_behavior_report(
            video_name=source_path.name,
            resolution=metadata['resolution'],
            total_frames=metadata['frame_count'],
            detection_model=self.model_name,
            pose_model=self._phase3_params['pose_model'],
            identity_summary=self.identity_manager.get_summary(),
            behavior_events=all_events,
            event_counts=event_counts,
            identity_mappings=identity_mappings,
            output_dir=report_dir
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Phase 3 Processing Complete")
        print(f"{'='*60}")
        print(f"Frames processed: {self.metrics.total_frames_processed}")
        print(f"Average FPS: {self.metrics.average_fps:.1f}")
        print(f"Unique tracks: {len(self.metrics.all_track_ids)}")
        print(f"Total behavior events: {len(all_events)}")
        if event_counts:
            for etype, count in sorted(event_counts.items()):
                print(f"  {etype}: {count}")
        print(f"Processing time: {format_time(self.metrics.total_processing_time)}")
        print(f"Output video: {output_path}")
        print(f"{'='*60}\n")
        
        return report
    
    def _draw_behavior_box(self, frame, box, track_id: int,
                            student_name: Optional[str],
                            active_events: dict,
                            head_state: dict):
        """Draw a behavior-annotated bounding box."""
        x1, y1, x2, y2 = [int(v) for v in box]
        
        # Determine primary color based on most important active event
        if 'possible_sleeping' in active_events:
            color = BEHAVIOR_COLORS['possible_sleeping']
        elif 'phone_usage' in active_events:
            color = BEHAVIOR_COLORS['phone_usage']
        elif 'raised_hand' in active_events:
            color = BEHAVIOR_COLORS['raised_hand']
        elif 'head_down' in active_events:
            color = BEHAVIOR_COLORS['head_down']
        else:
            orientation = head_state.get('orientation', '')
            if orientation == 'FORWARD':
                color = BEHAVIOR_COLORS['head_forward']
            elif orientation in ('LEFT', 'RIGHT'):
                color = BEHAVIOR_COLORS.get(f'head_{orientation.lower()}', get_color(track_id))
            else:
                color = get_color(track_id)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Build label lines
        lines = []
        if student_name:
            lines.append(student_name)
        else:
            lines.append(f"Track {track_id}")
        
        # Add behavior status
        if 'possible_sleeping' in active_events:
            lines.append('POSSIBLE SLEEPING')
        elif 'phone_usage' in active_events:
            lines.append('PHONE DETECTED')
        elif 'raised_hand' in active_events:
            lines.append('HAND RAISED')
        elif head_state.get('orientation') in ('DOWN', 'LEFT', 'RIGHT'):
            lines.append(f"Head: {head_state['orientation']}")
        
        # Draw labels above the box
        y_offset = y1
        for i, text in enumerate(lines):
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = y_offset - (len(lines) - i) * (th + baseline + 4)
            bg_color = color if i == 0 else (60, 60, 60)
            cv2.rectangle(frame, (x1, label_y - 2),
                          (x1 + tw + 4, label_y + th + baseline),
                          bg_color, -1)
            cv2.putText(frame, text, (x1 + 2, label_y + th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
