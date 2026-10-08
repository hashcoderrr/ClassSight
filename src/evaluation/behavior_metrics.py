"""Behavior analysis metrics and reporting for Phase 3.

Generates behavior reports with event counts, per-student events,
performance metrics, and temporal analysis.

Does NOT calculate engagement scores — that belongs to a later phase.
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import statistics


class BehaviorMetrics:
    """Collects and reports behavior analysis metrics."""
    
    def __init__(self):
        self.frame_times: list[float] = []
        self.behavior_times: list[float] = []
        self.frame_student_counts: list[int] = []
        self.all_track_ids: set[int] = set()
        self.total_frames_processed: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Behavior-specific counts
        self.phone_detections: int = 0
        self.hand_raises: int = 0
        self.head_down_frames: int = 0
        self.sleep_indicators: int = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def update_frame(self, frame_time: float, student_count: int,
                      track_ids: list[int]):
        self.frame_times.append(frame_time)
        self.frame_student_counts.append(student_count)
        self.total_frames_processed += 1
        for tid in track_ids:
            if tid != -1:
                self.all_track_ids.add(tid)
    
    def update_behavior(self, behavior_result: dict, behavior_time: float):
        self.behavior_times.append(behavior_time)
        
        for tid, phone in behavior_result.get('phone_results', {}).items():
            if phone.get('detected', False):
                self.phone_detections += 1
        
        for tid, hand in behavior_result.get('hand_results', {}).items():
            if hand.get('detected', False):
                self.hand_raises += 1
        
        for tid, head in behavior_result.get('head_results', {}).items():
            if head.get('orientation') == 'DOWN':
                self.head_down_frames += 1
        
        for tid, sleep in behavior_result.get('sleep_results', {}).items():
            if sleep.get('detected', False):
                self.sleep_indicators += 1
    
    @property
    def current_fps(self) -> float:
        if self.frame_times and self.frame_times[-1] > 0:
            return 1.0 / self.frame_times[-1]
        return 0.0
    
    @property
    def average_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg = statistics.mean(self.frame_times)
        return 1.0 / avg if avg > 0 else 0.0
    
    @property
    def average_behavior_time(self) -> float:
        return statistics.mean(self.behavior_times) if self.behavior_times else 0.0
    
    @property
    def total_processing_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return sum(self.frame_times)
    
    def generate_behavior_report(self,
                                  video_name: str,
                                  resolution: str,
                                  total_frames: int,
                                  detection_model: str,
                                  pose_model: str,
                                  identity_summary: dict,
                                  behavior_events: list,
                                  event_counts: dict,
                                  identity_mappings: dict,
                                  output_dir: str | Path,
                                  save_csv: bool = True) -> dict:
        """Generate and save the behavior analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Per-student events
        per_student = {}
        for event in behavior_events:
            key = event.student_id or f"track_{event.track_id}"
            if key not in per_student:
                per_student[key] = []
            per_student[key].append(event.to_dict())
        
        report = {
            'video': video_name,
            'detection_model': detection_model,
            'pose_model': pose_model,
            'resolution': resolution,
            'total_frames': total_frames,
            'processed_frames': self.total_frames_processed,
            'unique_tracks': len(self.all_track_ids),
            'identified_students': identity_summary.get('confirmed', 0),
            'unknown_tracks': identity_summary.get('unknown', 0),
            'total_behavior_events': len(behavior_events),
            'event_counts': event_counts,
            'phone_detection_frames': self.phone_detections,
            'hand_raise_frames': self.hand_raises,
            'head_down_frames': self.head_down_frames,
            'sleep_indicator_frames': self.sleep_indicators,
            'average_fps': round(self.average_fps, 2),
            'average_behavior_time_ms': round(self.average_behavior_time * 1000, 1),
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'processing_date': datetime.now().isoformat(),
            'per_student_events': per_student,
        }
        
        json_path = output_dir / f'behavior_report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"[INFO] Behavior report saved: {json_path}")
        
        if save_csv:
            csv_path = output_dir / f'behavior_report_{timestamp}.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in report.items():
                    if key not in ('per_student_events', 'event_counts'):
                        writer.writerow([key, value])
                # Write event counts
                for etype, count in event_counts.items():
                    writer.writerow([f'event_count_{etype}', count])
            print(f"[INFO] Behavior CSV saved: {csv_path}")
        
        return report
