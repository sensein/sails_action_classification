"""
Tracking Data Export Module
Collects and exports comprehensive tracking data to JSON format
"""

import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import defaultdict
import os
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and torch tensors"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class TrackingDataExporter:
    """Exports comprehensive tracking data to JSON"""

    def __init__(self):
        self.video_metadata = {}
        self.tracking_data = defaultdict(lambda: {
            'start_frame': None,
            'end_frame': None,
            'frames': {}
        })

    def set_video_metadata(self, input_path: str, total_frames: int, fps: float,
                          width: int, height: int):
        """Set video metadata"""
        self.video_metadata = {
            'input_path': input_path,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'export_timestamp': datetime.now().isoformat()
        }

    def add_frame_data(self, frame_number: int, detections: List[Dict],
                      person_assignments: Dict[int, int]):
        """Add data for a single frame"""

        # Individual tracking data
        for det_idx, track_id in person_assignments.items():
            if det_idx < len(detections):
                detection = detections[det_idx]

                # Initialize track data if first appearance
                if self.tracking_data[track_id]['start_frame'] is None:
                    self.tracking_data[track_id]['start_frame'] = frame_number

                # Update end frame
                self.tracking_data[track_id]['end_frame'] = frame_number

                # Store frame data
                frame_data = {
                    'bbox': detection['bbox'].tolist() if isinstance(detection['bbox'], np.ndarray) else detection['bbox'],
                    'keypoints': self._process_keypoints(detection['keypoints'])
                }

                self.tracking_data[track_id]['frames'][frame_number] = frame_data

    def _process_keypoints(self, keypoints) -> List[List[float]]:
        """Process keypoints to consistent format"""
        if torch.is_tensor(keypoints):
            kpts = keypoints.cpu().numpy()
        else:
            kpts = keypoints

        if len(kpts.shape) == 3:
            kpts = kpts[0]  # Remove batch dimension if present

        # Convert to list of [x, y, confidence] for each keypoint
        processed_kpts = []
        for i in range(len(kpts)):
            if len(kpts[i]) >= 3:
                processed_kpts.append([float(kpts[i][0]), float(kpts[i][1]), float(kpts[i][2])])
            elif len(kpts[i]) >= 2:
                processed_kpts.append([float(kpts[i][0]), float(kpts[i][1]), 0.0])

        return processed_kpts

    def finalize_data(self, processing_time: float):
        """Finalize data before export"""
        self.video_metadata['processing_time'] = processing_time

    def export_to_json(self, output_path: str):
        """Export all tracking data to JSON file"""

        # Prepare final data structure
        export_data = {
            'video_metadata': self.video_metadata,
            'tracking_results': dict(self.tracking_data),
            'export_info': {
                'total_tracks': len(self.tracking_data),
                'export_timestamp': datetime.now().isoformat()
            }
        }

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, cls=NumpyEncoder, indent=2)

        print(f"Tracking data exported to: {output_path}")
        print(f"Total tracks: {len(self.tracking_data)}")

class TrackingDataCollector:
    """Collects tracking data during processing"""

    def __init__(self):
        self.exporter = TrackingDataExporter()

    def collect_frame_data(self, frame_number: int, detections: List[Dict],
                          person_assignments: Dict[int, int]):
        """Collect data for current frame"""

        # Add to exporter
        self.exporter.add_frame_data(frame_number, detections, person_assignments)

    def set_video_info(self, input_path: str, total_frames: int, fps: float,
                      width: int, height: int):
        """Set video metadata"""
        self.exporter.set_video_metadata(input_path, total_frames, fps, width, height)

    def export_data(self, output_path: str, processing_time: float):
        """Export all collected data"""
        self.exporter.finalize_data(processing_time)
        self.exporter.export_to_json(output_path)