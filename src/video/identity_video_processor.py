"""Phase 2 video processor with face recognition and identity mapping."""

import cv2
import time
from pathlib import Path
from typing import Optional

from src.tracking.student_tracker import StudentTracker
from src.recognition.face_recognizer import FaceRecognizer
from src.recognition.identity_manager import IdentityManager
from src.evaluation.identity_metrics import IdentityMetrics
from src.utils.helpers import (
    ensure_dir, validate_video_path, get_color,
    draw_detection, draw_info_overlay, format_time
)


class IdentityVideoProcessor:
    """Video processor with YOLO detection, ByteTrack tracking, and face recognition.
    
    Pipeline per frame:
    1. YOLO + ByteTrack → person boxes + track IDs (Phase 1)
    2. Every N frames: Face detection + recognition within person crops
    3. IdentityManager: temporal confirmation of Track ID → Student ID
    4. Draw annotated overlays with identity info
    5. Write output video + generate identity report
    """
    
    def __init__(self, 
                 model_name: str = 'yolo11n.pt',
                 confidence: float = 0.4,
                 iou: float = 0.5,
                 frame_skip: int = 1,
                 show: bool = False,
                 # Phase 2 params
                 insightface_model: str = 'buffalo_l',
                 identities_file: str = 'data/identities/students.json',
                 embeddings_dir: str = 'data/identities/embeddings',
                 recognition_threshold: float = 0.4,
                 confirmation_frames: int = 3,
                 recognition_interval: int = 5,
                 min_face_size: int = 40,
                 retention_frames: int = 30):
        
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.frame_skip = max(1, frame_skip)
        self.show = show
        self.recognition_interval = max(1, recognition_interval)
        
        # Phase 1 components
        self.tracker = StudentTracker(
            model_name=model_name,
            confidence=confidence,
            iou=iou
        )
        
        # Phase 2 components
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
        self.metrics = IdentityMetrics()
    
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
        """Process a video with detection, tracking, and face recognition."""
        source_path = validate_video_path(source)
        
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {source_path}")
        
        metadata = self.get_video_metadata(cap)
        
        # Print header
        print(f"\n{'='*60}")
        print(f"ClassSight AI - Phase 2: Identity Video Processing")
        print(f"{'='*60}")
        print(f"Source: {source_path.name}")
        print(f"Resolution: {metadata['resolution']}")
        print(f"FPS: {metadata['fps']:.1f}")
        print(f"Frames: {metadata['frame_count']}")
        print(f"Duration: {format_time(metadata['duration'])}")
        print(f"Detection Model: {self.model_name}")
        print(f"Recognition Model: InsightFace ({self.recognizer.detector.model_name})")
        print(f"Enrolled Students: {self.recognizer.enrolled_count}")
        print(f"Recognition Interval: every {self.recognition_interval} frames")
        print(f"Confirmation Frames: {self.identity_manager.confirmation_frames}")
        print(f"{'='*60}\n")
        
        # Setup output
        if output_path is None:
            output_dir = ensure_dir(Path(__file__).parent.parent.parent / 'outputs' / 'videos')
            output_path = str(output_dir / f"{source_path.stem}_identity.mp4")
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
        processed_count = 0  # Count of frames actually processed (after skip)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Frame skipping
                if frame_number % self.frame_skip != 0:
                    out.write(frame)
                    continue
                
                processed_count += 1
                frame_start = time.time()
                
                # Phase 1: Track persons
                track_result = self.tracker.track(frame)
                
                # Phase 2: Face recognition (every N processed frames)
                run_recognition = (processed_count % self.recognition_interval == 0 
                                    and self.recognizer.enrolled_count > 0)
                
                if run_recognition and track_result['count'] > 0:
                    recog_start = time.time()
                    recog_results = self.recognizer.recognize_in_frame(
                        frame,
                        track_result['boxes'],
                        track_result['track_ids']
                    )
                    recog_time = time.time() - recog_start
                    
                    # Update identity manager with recognition results
                    for rr in recog_results:
                        self.identity_manager.update(
                            track_id=rr.track_id,
                            student_id=rr.student_id,
                            name=rr.name,
                            similarity=rr.similarity,
                            is_known=(rr.status == 'MATCHED'),
                            has_face=(rr.status != 'NO_FACE' and rr.status != 'SMALL_FACE')
                        )
                    
                    self.metrics.update_recognition(recog_results, recog_time)
                else:
                    # Between recognition intervals, just maintain existing state
                    for tid in track_result['track_ids']:
                        if tid != -1:
                            self.identity_manager.update_no_recognition(tid)
                
                frame_time = time.time() - frame_start
                
                # Update metrics
                self.metrics.update_frame(
                    frame_time=frame_time,
                    confidences=track_result['confidences'],
                    student_count=track_result['count'],
                    track_ids=track_result['track_ids']
                )
                
                # Draw detections with identity info
                for box, conf, tid in zip(
                    track_result['boxes'],
                    track_result['confidences'],
                    track_result['track_ids']
                ):
                    identity = self.identity_manager.get_identity(tid) if tid != -1 else None
                    
                    if identity and identity.confirmed_id:
                        # Confirmed student
                        color = get_color(tid)
                        label = f"{identity.confirmed_name}\nID:{tid} | {identity.last_similarity:.2f}"
                        self._draw_identity_box(frame, box, identity.confirmed_name,
                                                tid, identity.last_similarity, 
                                                identity.status, color)
                    elif identity and identity.status == 'PENDING':
                        color = (0, 255, 255)  # Yellow for pending
                        self._draw_identity_box(frame, box, '?',
                                                tid, identity.last_similarity,
                                                'PENDING', color)
                    elif identity and identity.status == 'UNKNOWN':
                        color = (0, 0, 255)  # Red for unknown
                        self._draw_identity_box(frame, box, 'UNKNOWN',
                                                tid, 0.0, 'UNKNOWN', color)
                    else:
                        color = get_color(tid if tid != -1 else 0)
                        label = f"ID: {tid} ({conf:.2f})" if tid != -1 else f"Person ({conf:.2f})"
                        draw_detection(frame, box, label, color)
                
                # Draw overlay
                id_summary = self.identity_manager.get_summary()
                info = {
                    'FPS': f"{self.metrics.current_fps:.1f}",
                    'Students': str(track_result['count']),
                    'Identified': f"{id_summary['confirmed']}/{len(self.identity_manager._states)}",
                    'Frame': f"{frame_number}/{metadata['frame_count']}"
                }
                draw_info_overlay(frame, info)
                
                out.write(frame)
                
                if self.show:
                    cv2.imshow('ClassSight AI - Identity', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] User pressed Q - stopping.")
                        break
                
                if frame_number % 100 == 0:
                    print(f"  Frame {frame_number}/{metadata['frame_count']} "
                          f"| FPS: {self.metrics.current_fps:.1f} "
                          f"| Students: {track_result['count']} "
                          f"| Identified: {id_summary['confirmed']}")
        
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
        
        # Generate identity report
        report_dir = ensure_dir(Path(__file__).parent.parent.parent / 'outputs' / 'reports')
        report = self.metrics.generate_identity_report(
            video_name=source_path.name,
            resolution=metadata['resolution'],
            total_frames=metadata['frame_count'],
            detection_model=self.model_name,
            recognition_model=f"InsightFace ({self.recognizer.detector.model_name})",
            identity_summary=self.identity_manager.get_summary(),
            identity_mappings=self.identity_manager.get_all_mappings(),
            output_dir=report_dir
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Phase 2 Processing Complete")
        print(f"{'='*60}")
        summary = self.identity_manager.get_summary()
        print(f"Frames processed: {self.metrics.total_frames_processed}")
        print(f"Average FPS: {self.metrics.average_fps:.1f}")
        print(f"Unique tracks: {len(self.metrics.all_track_ids)}")
        print(f"Confirmed students: {summary['confirmed']}")
        print(f"Unknown tracks: {summary['unknown']}")
        print(f"Pending: {summary['pending']}")
        print(f"Avg recognition score: {self.metrics.average_recognition_score:.4f}")
        print(f"Processing time: {format_time(self.metrics.total_processing_time)}")
        print(f"Output video: {output_path}")
        print(f"{'='*60}\n")
        
        return report
    
    def _draw_identity_box(self, frame, box, name: str,
                            track_id: int, similarity: float,
                            status: str, color: tuple):
        """Draw an identity-annotated bounding box."""
        x1, y1, x2, y2 = [int(v) for v in box]
        thickness = 2 if status != 'CONFIRMED' else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw name label at top
        display_name = name if name else 'UNKNOWN'
        (tw, th), baseline = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, display_name, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw track ID and match info below the name
        info_text = f"ID:{track_id}"
        if similarity > 0:
            info_text += f" | {similarity:.2f}"
        
        (tw2, th2), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = y1 - th - baseline - 5 - th2 - 5
        cv2.rectangle(frame, (x1, label_y - 3), (x1 + tw2, label_y + th2 + 3), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (x1, label_y + th2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
