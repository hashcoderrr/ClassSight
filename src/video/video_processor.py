import cv2
import time
from pathlib import Path
from typing import Optional

from src.tracking.student_tracker import StudentTracker
from src.evaluation.metrics import PerformanceMetrics
from src.utils.helpers import (
    ensure_dir, validate_video_path, get_color, 
    draw_detection, draw_info_overlay, format_time
)

class VideoProcessor:
    """Orchestrates video processing with YOLO detection and ByteTrack tracking."""
    
    def __init__(self, model_name: str = 'yolo11n.pt',
                 confidence: float = 0.4,
                 iou: float = 0.5,
                 frame_skip: int = 1,
                 show: bool = False):
        """
        Args:
            model_name: YOLO model name/path
            confidence: Detection confidence threshold
            iou: IoU threshold for NMS
            frame_skip: Process every Nth frame (1 = every frame)
            show: Whether to display the video in a window
        """
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.frame_skip = max(1, frame_skip)
        self.show = show
        
        self.tracker = StudentTracker(
            model_name=model_name,
            confidence=confidence,
            iou=iou
        )
        self.metrics = PerformanceMetrics()
    
    def get_video_metadata(self, cap: cv2.VideoCapture) -> dict:
        """Extract video metadata from a VideoCapture object."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }
    
    def process(self, source: str, output_path: Optional[str] = None) -> dict:
        """Process a video file with detection and tracking.
        
        Args:
            source: Path to the input video
            output_path: Path for the output video. If None, auto-generated.
            
        Returns:
            dict: The generated performance report
        """
        # Validate input
        source_path = validate_video_path(source)
        
        # Open video
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        
        # Get metadata
        metadata = self.get_video_metadata(cap)
        print(f"\n{'='*60}")
        print(f"ClassSight AI - Video Processing")
        print(f"{'='*60}")
        print(f"Source: {source_path.name}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"FPS: {metadata['fps']:.1f}")
        print(f"Frames: {metadata['frame_count']}")
        print(f"Duration: {format_time(metadata['duration'])}")
        print(f"Model: {self.model_name}")
        print(f"Confidence: {self.confidence}")
        print(f"IoU: {self.iou}")
        print(f"Frame Skip: {self.frame_skip}")
        print(f"{'='*60}\n")
        
        # Setup output
        if output_path is None:
            output_dir = ensure_dir(Path(__file__).parent.parent.parent / 'outputs' / 'videos')
            output_path = str(output_dir / f"{source_path.stem}_tracked.mp4")
        else:
            ensure_dir(Path(output_path).parent)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            metadata['fps'],
            (metadata['width'], metadata['height'])
        )
        
        if not out.isOpened():
            cap.release()
            raise IOError(f"Cannot create output video: {output_path}")
        
        # Process frames
        self.metrics.start()
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Frame skipping
                if frame_number % self.frame_skip != 0:
                    out.write(frame)  # Write unprocessed frame
                    continue
                
                # Track persons
                frame_start = time.time()
                track_result = self.tracker.track(frame)
                frame_time = time.time() - frame_start
                
                # Update metrics
                self.metrics.update(
                    frame_time=frame_time,
                    confidences=track_result['confidences'],
                    student_count=track_result['count'],
                    track_ids=track_result['track_ids'],
                    frame_number=frame_number
                )
                
                # Draw detections
                for box, conf, tid in zip(
                    track_result['boxes'],
                    track_result['confidences'],
                    track_result['track_ids']
                ):
                    color = get_color(tid if tid != -1 else 0)
                    label = f"ID: {tid} ({conf:.2f})" if tid != -1 else f"Student ({conf:.2f})"
                    draw_detection(frame, box, label, color)
                
                # Draw overlay info
                info = {
                    'FPS': f"{self.metrics.current_fps:.1f}",
                    'Students': str(track_result['count']),
                    'Unique IDs': str(track_result['unique_ids_total']),
                    'Frame': f"{frame_number}/{metadata['frame_count']}"
                }
                draw_info_overlay(frame, info)
                
                # Write frame
                out.write(frame)
                
                # Show if requested
                if self.show:
                    cv2.imshow('ClassSight AI', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] User pressed Q - stopping processing.")
                        break
                
                # Progress
                if frame_number % 100 == 0:
                    print(f"  Processed {frame_number}/{metadata['frame_count']} frames "
                          f"| FPS: {self.metrics.current_fps:.1f} "
                          f"| Students: {track_result['count']} "
                          f"| Unique IDs: {track_result['unique_ids_total']}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt - stopping processing.")
        
        finally:
            self.metrics.stop()
            if cap.isOpened():
                cap.release()
            if out.isOpened():
                out.release()
            if self.show:
                cv2.destroyAllWindows()
        
        # Generate report
        report_dir = ensure_dir(Path(__file__).parent.parent.parent / 'outputs' / 'reports')
        report = self.metrics.generate_report(
            video_name=source_path.name,
            resolution=metadata['resolution'],
            total_frames=metadata['frame_count'],
            model_name=self.model_name,
            output_dir=report_dir
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Processing Complete")
        print(f"{'='*60}")
        print(f"Frames processed: {self.metrics.total_frames_processed}")
        print(f"Average FPS: {self.metrics.average_fps:.1f}")
        print(f"Average confidence: {self.metrics.average_confidence:.4f}")
        print(f"Average visible students: {self.metrics.average_visible_students:.1f}")
        print(f"Max visible students: {self.metrics.max_visible_students}")
        print(f"Total unique track IDs: {self.metrics.unique_track_ids}")
        print(f"Processing time: {format_time(self.metrics.total_processing_time)}")
        print(f"Output video: {output_path}")
        print(f"{'='*60}\n")
        
        return report
