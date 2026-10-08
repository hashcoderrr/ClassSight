"""Central behavior analysis engine for classroom environments.

Orchestrates all behavior detection modules and combines results
with temporal event management. Delegates to independent modules
for each behavior type.
"""

import time
import numpy as np
from typing import Optional

from .pose_analyzer import PoseAnalyzer
from .phone_detector import PhoneDetector
from .hand_detector import HandDetector
from .head_pose import HeadPoseAnalyzer
from .sleep_detector import SleepDetector
from .event_manager import EventManager


class BehaviorEngine:
    """Orchestrates all behavior detection modules.
    
    Input: frame + tracked persons + identity mappings
    Output: per-frame behavior analysis + temporally confirmed events
    
    Components:
        PoseAnalyzer: YOLO-pose keypoint extraction
        PhoneDetector: YOLO cell phone detection + person association
        HandDetector: Raised hand geometry from pose landmarks
        HeadPoseAnalyzer: Head orientation classification
        SleepDetector: Sustained head-down posture indicator
        EventManager: Temporal confirmation of all behavior events
    """
    
    def __init__(self,
                 pose_model: str = 'yolo11n-pose.pt',
                 phone_model: str = 'yolo11n.pt',
                 phone_confidence: float = 0.3,
                 phone_class_id: int = 67,
                 phone_proximity: int = 100,
                 phone_confirmation_frames: int = 3,
                 hand_confirmation_frames: int = 2,
                 head_down_confirmation_frames: int = 5,
                 sleep_confirmation_seconds: float = 5.0,
                 min_event_duration: float = 0.5,
                 fps: float = 25.0):
        """Initialize the behavior engine.
        
        Args:
            pose_model: YOLO pose model name
            phone_model: YOLO model for phone detection
            phone_confidence: Phone detection confidence threshold
            phone_class_id: COCO class ID for cell phone
            phone_proximity: Max pixel distance for phone-person association
            phone_confirmation_frames: Frames to confirm phone usage
            hand_confirmation_frames: Frames to confirm raised hand
            head_down_confirmation_frames: Frames to confirm head-down event
            sleep_confirmation_seconds: Seconds of head-down for sleep indicator
            min_event_duration: Minimum event duration in seconds
            fps: Video FPS
        """
        self.fps = fps
        
        # Behavior modules
        self.pose_analyzer = PoseAnalyzer(model_name=pose_model)
        self.phone_detector = PhoneDetector(
            model_name=phone_model,
            confidence=phone_confidence,
            phone_class_id=phone_class_id,
            proximity_pixels=phone_proximity
        )
        self.hand_detector = HandDetector()
        self.head_pose = HeadPoseAnalyzer()
        self.sleep_detector = SleepDetector(
            confirmation_seconds=sleep_confirmation_seconds,
            fps=fps
        )
        
        # Event managers per behavior type (different confirmation thresholds)
        self.event_managers = {
            'phone_usage': EventManager(
                confirmation_frames=phone_confirmation_frames,
                min_duration=min_event_duration,
                fps=fps
            ),
            'raised_hand': EventManager(
                confirmation_frames=hand_confirmation_frames,
                min_duration=min_event_duration,
                fps=fps
            ),
            'head_down': EventManager(
                confirmation_frames=head_down_confirmation_frames,
                min_duration=min_event_duration,
                fps=fps
            ),
            'head_left': EventManager(
                confirmation_frames=head_down_confirmation_frames,
                min_duration=min_event_duration,
                fps=fps
            ),
            'head_right': EventManager(
                confirmation_frames=head_down_confirmation_frames,
                min_duration=min_event_duration,
                fps=fps
            ),
            'possible_sleeping': EventManager(
                confirmation_frames=int(sleep_confirmation_seconds * fps),
                min_duration=sleep_confirmation_seconds,
                fps=fps
            ),
        }
    
    def process_frame(self, frame: np.ndarray,
                       track_result: dict,
                       identity_mappings: Optional[dict] = None,
                       frame_number: int = 0) -> dict:
        """Run all behavior analysis on a single frame.
        
        Args:
            frame: BGR frame
            track_result: Phase 1 tracking result with keys:
                'boxes', 'confidences', 'track_ids', 'count'
            identity_mappings: Optional Phase 2 identity mappings
                {track_id: {'student_id': str, 'name': str, 'status': str}}
            frame_number: Current frame number (1-indexed)
        
        Returns:
            dict with analysis results:
                'pose_results': {tid: {...}},
                'phone_results': {tid: {...}},
                'hand_results': {tid: {...}},
                'head_results': {tid: {...}},
                'sleep_results': {tid: {...}},
                'active_events': {tid: {event_type: True}},
                'analysis_time': float
        """
        if track_result['count'] == 0:
            return {
                'pose_results': {},
                'phone_results': {},
                'hand_results': {},
                'head_results': {},
                'sleep_results': {},
                'active_events': {},
                'analysis_time': 0.0
            }
        
        start = time.time()
        
        boxes = track_result['boxes']
        tids = track_result['track_ids']
        
        # 1. Pose estimation
        pose_results = self.pose_analyzer.analyze(frame, boxes, tids)
        
        # 2. Phone detection
        phone_results = self.phone_detector.detect(frame, boxes, tids, pose_results)
        
        # 3. Raised hand detection (from pose)
        hand_results = self.hand_detector.detect(pose_results)
        
        # 4. Head orientation (from pose)
        head_results = self.head_pose.analyze(pose_results)
        
        # 5. Sleep indicators (from pose + head orientation)
        sleep_results = self.sleep_detector.detect(pose_results, head_results, tids)
        
        # 6. Update event managers with raw detections
        for tid in tids:
            if tid == -1:
                continue
            
            # Phone events
            phone = phone_results.get(tid, {})
            self.event_managers['phone_usage'].update(
                tid, 'phone_usage',
                phone.get('detected', False),
                phone.get('confidence', 0.0),
                frame_number
            )
            
            # Hand raise events
            hand = hand_results.get(tid, {})
            self.event_managers['raised_hand'].update(
                tid, 'raised_hand',
                hand.get('detected', False),
                hand.get('confidence', 0.0),
                frame_number
            )
            
            # Head orientation events
            head = head_results.get(tid, {})
            orientation = head.get('orientation', 'UNKNOWN')
            head_conf = head.get('confidence', 0.0)
            
            self.event_managers['head_down'].update(
                tid, 'head_down',
                orientation == 'DOWN',
                head_conf,
                frame_number
            )
            self.event_managers['head_left'].update(
                tid, 'head_left',
                orientation == 'LEFT',
                head_conf,
                frame_number
            )
            self.event_managers['head_right'].update(
                tid, 'head_right',
                orientation == 'RIGHT',
                head_conf,
                frame_number
            )
            
            # Sleeping events
            sleep = sleep_results.get(tid, {})
            self.event_managers['possible_sleeping'].update(
                tid, 'possible_sleeping',
                sleep.get('detected', False),
                sleep.get('confidence', 0.0),
                frame_number
            )
        
        # Collect active events across all managers
        active_events: dict[int, dict[str, bool]] = {}
        for etype, manager in self.event_managers.items():
            for tid_events in [manager.get_all_active_events()]:
                for tid, events in tid_events.items():
                    if tid not in active_events:
                        active_events[tid] = {}
                    active_events[tid].update(events)
        
        analysis_time = time.time() - start
        
        return {
            'pose_results': pose_results,
            'phone_results': phone_results,
            'hand_results': hand_results,
            'head_results': head_results,
            'sleep_results': sleep_results,
            'active_events': active_events,
            'analysis_time': analysis_time
        }
    
    def get_confirmed_events(self):
        """Get all confirmed events from all managers."""
        all_events = []
        for manager in self.event_managers.values():
            all_events.extend(manager.get_confirmed_events())
        return all_events
    
    def finalize(self, identity_mappings=None):
        """Finalize all pending events at video end."""
        all_events = []
        for manager in self.event_managers.values():
            all_events.extend(manager.finalize(identity_mappings))
        return all_events
    
    def get_event_counts(self):
        """Get aggregate event counts across all managers."""
        counts = {}
        for etype, manager in self.event_managers.items():
            c = manager.get_event_counts()
            for k, v in c.items():
                counts[k] = counts.get(k, 0) + v
        return counts
    
    def reset(self):
        """Reset all modules and event managers."""
        self.sleep_detector.reset()
        for manager in self.event_managers.values():
            manager.reset()
