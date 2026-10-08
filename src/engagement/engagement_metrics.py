"""Engagement analytics reporting and visualization.

Generates JSON/CSV reports and optional matplotlib visualizations
for Phase 4 engagement analysis results.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional


class EngagementReporter:
    """Generates engagement analysis reports and visualizations."""
    
    def generate_report(self,
                         analysis_result: dict,
                         video_name: str = '',
                         resolution: str = '',
                         total_frames: int = 0,
                         fps: float = 0.0,
                         session_duration: float = 0.0,
                         output_dir: str | Path = 'outputs/reports',
                         save_csv: bool = True,
                         save_plots: bool = True) -> dict:
        """Generate and save the engagement analysis report.
        
        Args:
            analysis_result: Output from EngagementEngine.analyze()
            video_name: Source video filename
            resolution: Video resolution string
            total_frames: Total frame count
            fps: Video FPS
            session_duration: Total session duration in seconds
            output_dir: Directory for report files
            save_csv: Whether to save CSV report
            save_plots: Whether to generate visualization plots
        
        Returns:
            Complete report dict (also saved as JSON)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build report structure
        report = {
            'session': {
                'video': video_name,
                'duration': round(session_duration, 2),
                'total_frames': total_frames,
                'fps': round(fps, 1),
                'resolution': resolution,
                'processing_date': datetime.now().isoformat(),
            },
            'model': analysis_result.get('model', {}),
            'classroom': analysis_result.get('classroom', {}),
            'students': self._serialize_students(
                analysis_result.get('students', {})
            ),
            'timeline': analysis_result.get('timeline', []),
        }
        
        # Save JSON
        json_path = output_dir / f'engagement_report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        print(f"[INFO] Engagement report saved: {json_path}")
        
        # Save CSV
        if save_csv:
            csv_path = output_dir / f'engagement_report_{timestamp}.csv'
            self._save_csv(report, csv_path)
            print(f"[INFO] Engagement CSV saved: {csv_path}")
        
        # Generate visualizations
        if save_plots:
            self._generate_plots(
                report, output_dir, timestamp
            )
        
        return report
    
    def _serialize_students(self, students: dict) -> dict:
        """Serialize student results for JSON output."""
        serialized = {}
        for key, result in students.items():
            entry = {
                'track_id': result.get('track_id'),
                'student_id': result.get('student_id'),
                'engagement_indicator': result.get('engagement_indicator'),
                'engagement_level': result.get('engagement_level'),
                'data_quality': result.get('data_quality'),
                'contributing_factors': result.get('contributing_factors', []),
            }
            # Include key feature values (not all internal details)
            features = result.get('features', {})
            entry['features'] = {
                'visible_duration': features.get('visible_duration', 0.0),
                'forward_ratio': features.get('forward_ratio', 0.0),
                'head_down_ratio': features.get('head_down_ratio', 0.0),
                'phone_ratio': features.get('phone_ratio', 0.0),
                'sleep_ratio': features.get('sleep_ratio', 0.0),
                'raised_hand_count': features.get('raised_hand_count', 0),
                'raised_hand_rate': features.get('raised_hand_rate', 0.0),
                'total_event_count': features.get('total_event_count', 0),
            }
            serialized[key] = entry
        return serialized
    
    def _save_csv(self, report: dict, csv_path: Path):
        """Save engagement report as CSV."""
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Session info
            writer.writerow(['Section', 'Metric', 'Value'])
            session = report.get('session', {})
            for key, value in session.items():
                writer.writerow(['Session', key, value])
            
            # Classroom metrics
            classroom = report.get('classroom', {})
            for key, value in classroom.items():
                if isinstance(value, dict):
                    for k2, v2 in value.items():
                        writer.writerow(['Classroom', f'{key}.{k2}', v2])
                else:
                    writer.writerow(['Classroom', key, value])
            
            # Student engagement
            writer.writerow([])
            writer.writerow([
                'Student/Track', 'Engagement', 'Level',
                'Data Quality', 'Forward%', 'HeadDown%',
                'Phone%', 'HandRaise#', 'Events'
            ])
            for key, student in report.get('students', {}).items():
                features = student.get('features', {})
                writer.writerow([
                    key,
                    student.get('engagement_indicator', ''),
                    student.get('engagement_level', ''),
                    student.get('data_quality', ''),
                    round(features.get('forward_ratio', 0) * 100, 1),
                    round(features.get('head_down_ratio', 0) * 100, 1),
                    round(features.get('phone_ratio', 0) * 100, 1),
                    features.get('raised_hand_count', 0),
                    features.get('total_event_count', 0),
                ])
            
            # Timeline
            if report.get('timeline'):
                writer.writerow([])
                writer.writerow([
                    'Window Start', 'Window End',
                    'Engagement', 'Students', 'Events'
                ])
                for window in report['timeline']:
                    writer.writerow([
                        window['start'], window['end'],
                        window['engagement_indicator'],
                        window['student_count'],
                        window['events'],
                    ])
    
    def _generate_plots(self, report: dict,
                         output_dir: Path, timestamp: str):
        """Generate matplotlib visualization plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARNING] matplotlib not available, skipping plots.")
            return
        
        self._plot_timeline(report, output_dir, timestamp, plt)
        self._plot_engagement_distribution(report, output_dir, timestamp, plt)
        self._plot_behavior_breakdown(report, output_dir, timestamp, plt)
    
    def _plot_timeline(self, report, output_dir, timestamp, plt):
        """Plot engagement timeline."""
        timeline = report.get('timeline', [])
        if not timeline:
            return
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        times = [(w['start'] + w['end']) / 2 for w in timeline]
        scores = [w['engagement_indicator'] for w in timeline]
        
        ax.plot(times, scores, 'b-o', markersize=4, linewidth=1.5)
        ax.fill_between(times, scores, alpha=0.15, color='blue')
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Engagement Indicator')
        ax.set_title('Classroom Engagement Timeline')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Add level zones
        ax.axhspan(70, 100, alpha=0.05, color='green', label='HIGH')
        ax.axhspan(40, 70, alpha=0.05, color='yellow', label='MODERATE')
        ax.axhspan(0, 40, alpha=0.05, color='red', label='LOW')
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        path = output_dir / f'engagement_timeline_{timestamp}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Timeline plot saved: {path}")
    
    def _plot_engagement_distribution(self, report, output_dir, timestamp, plt):
        """Plot student engagement score distribution."""
        students = report.get('students', {})
        scores = [
            s['engagement_indicator']
            for s in students.values()
            if s.get('engagement_indicator') is not None
        ]
        if not scores:
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        bins = list(range(0, 105, 10))
        ax.hist(scores, bins=bins, color='steelblue', edgecolor='white',
                alpha=0.8)
        
        ax.axvline(x=40, color='orange', linestyle='--', alpha=0.7,
                   label='LOW/MODERATE boundary')
        ax.axvline(x=70, color='green', linestyle='--', alpha=0.7,
                   label='MODERATE/HIGH boundary')
        
        ax.set_xlabel('Engagement Indicator')
        ax.set_ylabel('Number of Students')
        ax.set_title('Student Engagement Distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = output_dir / f'engagement_distribution_{timestamp}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Distribution plot saved: {path}")
    
    def _plot_behavior_breakdown(self, report, output_dir, timestamp, plt):
        """Plot behavior duration breakdown."""
        breakdown = report.get('classroom', {}).get(
            'behavior_breakdown', {}
        )
        if not breakdown:
            return
        
        # Filter to types with duration data
        labels = []
        durations = []
        colors_list = [
            '#e74c3c', '#2ecc71', '#e67e22',
            '#3498db', '#3498db', '#c0392b'
        ]
        type_order = [
            'phone_usage', 'raised_hand', 'head_down',
            'head_left', 'head_right', 'possible_sleeping'
        ]
        
        for i, btype in enumerate(type_order):
            if btype in breakdown:
                dur = breakdown[btype].get('total_duration', 0)
                if dur > 0:
                    labels.append(btype.replace('_', ' ').title())
                    durations.append(dur)
        
        if not labels:
            return
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(labels, durations, color=colors_list[:len(labels)],
                       alpha=0.8)
        
        for bar, dur in zip(bars, durations):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{dur:.1f}s', va='center', fontsize=9)
        
        ax.set_xlabel('Total Duration (seconds)')
        ax.set_title('Behavior Duration Breakdown (Aggregate)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        path = output_dir / f'behavior_distribution_{timestamp}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Behavior breakdown plot saved: {path}")
