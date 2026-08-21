import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import statistics

class PerformanceMetrics:
    """Collects and reports performance metrics for video processing."""
    
    def __init__(self):
        self.frame_times: list[float] = []  # per-frame processing times
        self.frame_confidences: list[list[float]] = []  # confidences per frame
        self.frame_student_counts: list[int] = []  # students per frame
        self.all_track_ids: set[int] = set()
        self.total_frames_processed: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.low_confidence_log: list[dict] = []  # frames with low-confidence detections
    
    def start(self):
        """Mark the start of processing."""
        self.start_time = time.time()
    
    def stop(self):
        """Mark the end of processing."""
        self.end_time = time.time()
    
    def update(self, frame_time: float, confidences: list[float], 
               student_count: int, track_ids: list[int],
               frame_number: int = 0, low_conf_threshold: float = 0.3):
        """Update metrics with data from a processed frame."""
        self.frame_times.append(frame_time)
        self.frame_confidences.append(confidences)
        self.frame_student_counts.append(student_count)
        self.total_frames_processed += 1
        
        for tid in track_ids:
            if tid != -1:
                self.all_track_ids.add(tid)
        
        # Log low confidence detections
        low_confs = [c for c in confidences if c < low_conf_threshold]
        if low_confs:
            self.low_confidence_log.append({
                'frame': frame_number,
                'low_confidences': low_confs,
                'count': len(low_confs)
            })
    
    @property
    def current_fps(self) -> float:
        if self.frame_times and self.frame_times[-1] > 0:
            return 1.0 / self.frame_times[-1]
        return 0.0
    
    @property
    def average_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_time = statistics.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    @property
    def average_confidence(self) -> float:
        all_confs = [c for frame_confs in self.frame_confidences for c in frame_confs]
        return statistics.mean(all_confs) if all_confs else 0.0
    
    @property
    def average_visible_students(self) -> float:
        return statistics.mean(self.frame_student_counts) if self.frame_student_counts else 0.0
    
    @property
    def max_visible_students(self) -> int:
        return max(self.frame_student_counts) if self.frame_student_counts else 0
    
    @property
    def unique_track_ids(self) -> int:
        return len(self.all_track_ids)
    
    @property
    def total_processing_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return sum(self.frame_times)
    
    def generate_report(self, video_name: str, resolution: str, 
                         total_frames: int, model_name: str,
                         output_dir: str | Path,
                         save_csv: bool = True) -> dict:
        """Generate and save a summary report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = {
            'video': video_name,
            'model': model_name,
            'resolution': resolution,
            'total_frames': total_frames,
            'processed_frames': self.total_frames_processed,
            'average_fps': round(self.average_fps, 2),
            'average_confidence': round(self.average_confidence, 4),
            'average_visible_students': round(self.average_visible_students, 1),
            'max_visible_students': self.max_visible_students,
            'unique_track_ids': self.unique_track_ids,
            'total_processing_time_seconds': round(self.total_processing_time, 2),
            'processing_date': datetime.now().isoformat(),
            'low_confidence_frames': len(self.low_confidence_log)
        }
        
        # Save JSON
        json_path = output_dir / f'report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"[INFO] Report saved: {json_path}")
        
        # Save CSV
        if save_csv:
            csv_path = output_dir / f'report_{timestamp}.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for key, value in report.items():
                    writer.writerow([key, value])
            print(f"[INFO] CSV report saved: {csv_path}")
        
        return report
