"""Identity recognition metrics and reporting for Phase 2."""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import statistics


class IdentityMetrics:
    """Collects and reports identity recognition metrics.
    
    Extends Phase 1 performance metrics with face recognition statistics.
    """
    
    def __init__(self):
        self.frame_times: list[float] = []
        self.recognition_times: list[float] = []
        self.frame_student_counts: list[int] = []
        self.frame_confidences: list[list[float]] = []
        self.all_track_ids: set[int] = set()
        self.total_frames_processed: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Identity-specific metrics
        self.recognition_attempts: int = 0
        self.faces_detected: int = 0
        self.confirmed_matches: int = 0
        self.unknown_detections: int = 0
        self.no_face_detections: int = 0
        self.recognition_scores: list[float] = []
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
    
    def update_frame(self, frame_time: float, confidences: list[float],
                      student_count: int, track_ids: list[int]):
        """Update with per-frame detection data (same as Phase 1)."""
        self.frame_times.append(frame_time)
        self.frame_confidences.append(confidences)
        self.frame_student_counts.append(student_count)
        self.total_frames_processed += 1
        for tid in track_ids:
            if tid != -1:
                self.all_track_ids.add(tid)
    
    def update_recognition(self, recognition_results: list, recognition_time: float):
        """Update with face recognition results for a frame."""
        self.recognition_times.append(recognition_time)
        
        for result in recognition_results:
            self.recognition_attempts += 1
            
            if result.status == 'MATCHED':
                self.faces_detected += 1
                self.confirmed_matches += 1
                self.recognition_scores.append(result.similarity)
            elif result.status == 'UNKNOWN':
                self.faces_detected += 1
                self.unknown_detections += 1
                if result.similarity > 0:
                    self.recognition_scores.append(result.similarity)
            elif result.status == 'NO_FACE':
                self.no_face_detections += 1
            elif result.status == 'SMALL_FACE':
                self.no_face_detections += 1
    
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
    def average_confidence(self) -> float:
        all_c = [c for fc in self.frame_confidences for c in fc]
        return statistics.mean(all_c) if all_c else 0.0
    
    @property
    def average_recognition_score(self) -> float:
        return statistics.mean(self.recognition_scores) if self.recognition_scores else 0.0
    
    @property
    def average_recognition_time(self) -> float:
        return statistics.mean(self.recognition_times) if self.recognition_times else 0.0
    
    @property
    def total_processing_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return sum(self.frame_times)
    
    def generate_identity_report(self, video_name: str, resolution: str,
                                  total_frames: int, detection_model: str,
                                  recognition_model: str,
                                  identity_summary: dict,
                                  identity_mappings: dict,
                                  output_dir: str | Path,
                                  save_csv: bool = True) -> dict:
        """Generate and save the Phase 2 identity report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build serializable mappings
        serializable_mappings = {}
        for tid, info in identity_mappings.items():
            serializable_mappings[str(tid)] = {
                'student_id': info.get('student_id'),
                'name': info.get('name'),
                'status': info.get('status'),
                'best_similarity': round(info.get('best_similarity', 0), 4)
            }
        
        report = {
            'video': video_name,
            'detection_model': detection_model,
            'recognition_model': recognition_model,
            'resolution': resolution,
            'total_frames': total_frames,
            'processed_frames': self.total_frames_processed,
            'unique_tracks': len(self.all_track_ids),
            'confirmed_students': identity_summary.get('confirmed', 0),
            'unknown_tracks': identity_summary.get('unknown', 0),
            'pending_tracks': identity_summary.get('pending', 0),
            'no_face_tracks': identity_summary.get('no_face', 0),
            'recognition_attempts': self.recognition_attempts,
            'faces_detected': self.faces_detected,
            'confirmed_matches': self.confirmed_matches,
            'unknown_detections': self.unknown_detections,
            'average_fps': round(self.average_fps, 2),
            'average_detection_confidence': round(self.average_confidence, 4),
            'average_recognition_score': round(self.average_recognition_score, 4),
            'average_recognition_time_ms': round(self.average_recognition_time * 1000, 1),
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'processing_date': datetime.now().isoformat(),
            'identity_mappings': serializable_mappings
        }
        
        json_path = output_dir / f'identity_report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"[INFO] Identity report saved: {json_path}")
        
        if save_csv:
            csv_path = output_dir / f'identity_report_{timestamp}.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in report.items():
                    if key != 'identity_mappings':
                        writer.writerow([key, value])
            print(f"[INFO] Identity CSV saved: {csv_path}")
        
        return report
