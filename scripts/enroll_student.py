#!/usr/bin/env python3
"""Enroll a student for face recognition.

Usage:
    python scripts/enroll_student.py --student-id STU001 --name "Student 001" --images img1.jpg img2.jpg
    python scripts/enroll_student.py --student-id STU001 --name "Student 001" --student-dir data/students/student_001/
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.face_detector import FaceDetector
from src.recognition.embedding_store import EmbeddingStore
from config import get_config


def parse_args():
    config = get_config()
    parser = argparse.ArgumentParser(
        description='ClassSight AI - Student Enrollment for Face Recognition'
    )
    parser.add_argument('--student-id', type=str, required=True, help='Unique student ID (e.g., STU001)')
    parser.add_argument('--name', type=str, required=True, help='Student display name')
    parser.add_argument('--images', type=str, nargs='+', default=None, help='Paths to face images')
    parser.add_argument('--student-dir', type=str, default=None, help='Directory containing face images')
    parser.add_argument('--list', action='store_true', help='List all enrolled students')
    parser.add_argument('--remove', action='store_true', help='Remove enrolled student')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()
    
    try:
        store = EmbeddingStore(
            identities_file=config.IDENTITIES_FILE,
            embeddings_dir=config.EMBEDDINGS_DIR
        )
        
        if args.list:
            students = store.list_students()
            if not students:
                print("No students enrolled.")
            else:
                print(f"\nEnrolled Students ({len(students)}):")
                print(f"{'ID':<12} {'Name':<25} {'Images':<8} {'Created'}")
                print('-' * 65)
                for s in students:
                    print(f"{s['student_id']:<12} {s['name']:<25} {s['num_images']:<8} {s['created_at'][:19]}")
            return
        
        if args.remove:
            if store.remove(args.student_id):
                print(f"[INFO] Removed student: {args.student_id}")
            else:
                print(f"[ERROR] Student not found: {args.student_id}")
            return
        
        # Collect images
        image_paths = []
        if args.images:
            image_paths = [Path(p) for p in args.images]
        elif args.student_dir:
            sdir = Path(args.student_dir)
            if not sdir.exists():
                print(f"[ERROR] Directory not found: {sdir}")
                sys.exit(1)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(sdir.glob(ext))
        
        if not image_paths:
            print("[ERROR] No images provided. Use --images or --student-dir.")
            sys.exit(1)
        
        print(f"\n[INFO] Enrolling student: {args.name} ({args.student_id})")
        print(f"[INFO] Processing {len(image_paths)} image(s)...")
        
        detector = FaceDetector(model_name=config.INSIGHTFACE_MODEL)
        
        result = store.enroll(
            student_id=args.student_id,
            name=args.name,
            images=image_paths,
            face_detector=detector
        )
        
        print(f"\n{'='*50}")
        print(f"Enrollment Result")
        print(f"{'='*50}")
        print(f"Success: {result['success']}")
        print(f"Student ID: {result['student_id']}")
        print(f"Name: {result['name']}")
        print(f"Images processed: {result['num_images_processed']}")
        print(f"Valid embeddings: {result['num_valid_embeddings']}")
        print(f"Message: {result['message']}")
        
        if result['rejected_images']:
            print(f"\nRejected images:")
            for rej in result['rejected_images']:
                print(f"  {rej['path']}: {rej['reason']}")
        
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
