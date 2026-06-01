"""
Tracking Data Export Module - Updated
Collects and exports tracking data to JSON and HDF5 formats
"""

import json
import numpy as np
import torch
import h5py
from typing import Dict, List, Any, Optional
from collections import defaultdict
import os
from datetime import datetime
from pathlib import Path


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

                # Store frame data with bbox including confidence
                bbox = detection['bbox'].tolist() if isinstance(detection['bbox'], np.ndarray) else detection['bbox']
                confidence = detection.get('confidence', 1.0)
                
                # Ensure bbox has confidence as 5th element
                if len(bbox) < 5:
                    bbox = list(bbox) + [confidence]
                
                frame_data = {
                    'bbox': bbox,  # [x1, y1, x2, y2, confidence]
                    'keypoints': self._process_keypoints(detection['keypoints'])
                }

                self.tracking_data[track_id]['frames'][frame_number] = frame_data

    def remove_track(self, track_id: int) -> bool:
        """
        Remove a track from collected data (for post-hoc filtering)
        
        Returns:
            bool: True if track was removed, False if track didn't exist
        """
        if track_id in self.tracking_data:
            del self.tracking_data[track_id]
            return True
        return False

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
        """Export all tracking data to JSON file with frame-level view"""
        
        # Ensure .json extension
        if not output_path.endswith('.json'):
            output_path = output_path + '.json'

        # Build frame-level data structure
        frame_level_data = {}
        
        for track_id, track_data in self.tracking_data.items():
            for frame_num, frame_data in track_data['frames'].items():
                frame_num_str = str(frame_num)
                
                if frame_num_str not in frame_level_data:
                    frame_level_data[frame_num_str] = {
                        'frame_number': frame_num,
                        'detections': []
                    }
                
                # Add detection with all required fields
                detection = {
                    'track_id': int(track_id),
                    'bbox': frame_data['bbox'],  # [x1, y1, x2, y2, confidence]
                    'confidence': float(frame_data['bbox'][4]) if len(frame_data['bbox']) > 4 else 1.0,
                    'keypoints': frame_data['keypoints']  
                }
                
                frame_level_data[frame_num_str]['detections'].append(detection)
        
        # Sort detections by track_id within each frame
        for frame_data in frame_level_data.values():
            frame_data['detections'].sort(key=lambda x: x['track_id'])

        # Prepare final data structure
        export_data = {
            'video_metadata': self.video_metadata,
            'frame_data': frame_level_data,
            'track_summary': dict(self.tracking_data),  # track-centric view
            'export_info': {
                'total_tracks': len(self.tracking_data),
                'total_frames': len(frame_level_data),
                'export_timestamp': datetime.now().isoformat()
            }
        }

        # Write to JSON file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, cls=NumpyEncoder, indent=2)

        print(f"Tracking data exported to: {output_path}")
        print(f"Total tracks: {len(self.tracking_data)}")
        print(f"Total frames with detections: {len(frame_level_data)}")

    def export_tracks_to_hdf5(self, output_dir: str, person_profiles: Dict[int, Dict]):
        """
        Export each track to its own HDF5 file containing:
        - Per-frame bboxes
        - Per-frame keypoints
        - Averaged embeddings (face, upper, lower)

        Only exports tracks that exist in person_profiles ( passed post-hoc filtering)

        Args:
            output_dir: Directory to save HDF5 files
            person_profiles: Dict mapping track_id to profile with averaged embeddings
        """
        output_path = Path(output_dir)
        
        # If output_path exists as a file, remove it first
        if output_path.exists() and output_path.is_file():
            output_path.unlink()
        
        # Now create the directory
        output_path.mkdir(parents=True, exist_ok=True)

        num_exported = 0
        num_skipped = 0
        
        for track_id, track_data in self.tracking_data.items():
            # Skip tracks that were filtered out in post-hoc filtering
            if track_id not in person_profiles:
                num_skipped += 1
                continue
                
            if not track_data['frames']:
                continue

            # Create HDF5 file for this track
            h5_filename = output_path / f"track_{track_id:04d}.h5"

            try:
                with h5py.File(h5_filename, 'w') as f:
                    # Store metadata
                    metadata_grp = f.create_group('metadata')
                    metadata_grp.attrs['track_id'] = track_id
                    metadata_grp.attrs['start_frame'] = track_data['start_frame']
                    metadata_grp.attrs['end_frame'] = track_data['end_frame']
                    metadata_grp.attrs['num_frames'] = len(track_data['frames'])
                    metadata_grp.attrs['export_timestamp'] = datetime.now().isoformat()

                    # Add video metadata
                    if self.video_metadata:
                        metadata_grp.attrs['video_fps'] = self.video_metadata.get('fps', 0)
                        metadata_grp.attrs['video_width'] = self.video_metadata.get('width', 0)
                        metadata_grp.attrs['video_height'] = self.video_metadata.get('height', 0)

                    # Store per-frame data
                    frames_grp = f.create_group('frames')
                    frame_numbers = sorted(track_data['frames'].keys())

                    for frame_num in frame_numbers:
                        frame_data = track_data['frames'][frame_num]
                        frame_grp = frames_grp.create_group(f'frame_{frame_num:06d}')

                        # Store bbox (first 4 elements only for HDF5)
                        bbox = np.array(frame_data['bbox'][:4], dtype=np.float32)
                        frame_grp.create_dataset('bbox', data=bbox, compression='gzip', compression_opts=4)
                        
                        # Store confidence separately
                        confidence = frame_data['bbox'][4] if len(frame_data['bbox']) > 4 else 1.0
                        frame_grp.attrs['confidence'] = float(confidence)

                        # Store keypoints
                        keypoints = np.array(frame_data['keypoints'], dtype=np.float32)
                        frame_grp.create_dataset('keypoints', data=keypoints, compression='gzip', compression_opts=4)

                    # Store averaged embeddings from person_profiles
                    profile = person_profiles[track_id]
                    embeddings_grp = f.create_group('embeddings')

                    # Store face embedding if available
                    if profile.get('face_feature') is not None:
                        face_emb = np.array(profile['face_feature'], dtype=np.float32)
                        embeddings_grp.create_dataset('face_feature', data=face_emb, compression='gzip', compression_opts=4)

                    # Store upper body embedding if available
                    if profile.get('upper_feature') is not None:
                        upper_emb = np.array(profile['upper_feature'], dtype=np.float32)
                        embeddings_grp.create_dataset('upper_feature', data=upper_emb, compression='gzip', compression_opts=4)

                    # Store lower body embedding if available
                    if profile.get('lower_feature') is not None:
                        lower_emb = np.array(profile['lower_feature'], dtype=np.float32)
                        embeddings_grp.create_dataset('lower_feature', data=lower_emb, compression='gzip', compression_opts=4)

                    # Store profile metadata
                    embeddings_grp.attrs['creation_frame'] = profile.get('creation_frame', -1)

                num_exported += 1

            except Exception as e:
                print(f"Error exporting track {track_id} to HDF5: {e}")
                continue

        print(f"Exported {num_exported} tracks to HDF5 files in: {output_path}")
        if num_skipped > 0:
            print(f"Skipped {num_skipped} filtered tracks (removed by post-hoc filtering)")
        return num_exported


class TrackingDataCollector:
    """Collects tracking data during processing"""

    def __init__(self, output_path: Optional[str] = None, enable_hdf5: bool = False):
        self.exporter = TrackingDataExporter()
        self.output_path = output_path
        self.enable_hdf5 = enable_hdf5

    def collect_frame_data(self, frame_number: int, detections: List[Dict],
                          person_assignments: Dict[int, int]):
        """Collect data for current frame"""
        self.exporter.add_frame_data(frame_number, detections, person_assignments)

    def set_video_info(self, input_path: str, total_frames: int, fps: float,
                      width: int, height: int):
        """Set video metadata"""
        self.exporter.set_video_metadata(input_path, total_frames, fps, width, height)

    def remove_track(self, track_id: int):
        """Remove a track from collected data (for post-hoc filtering)"""
        if self.exporter.remove_track(track_id):
            print(f"  Removed track {track_id} from export data")

    def export_data(self, output_path: str, processing_time: float, 
                   person_profiles: Optional[Dict[int, Dict]] = None):
        """
        Export all collected data

        Args:
            output_path: Path for JSON/HDF5 export
            processing_time: Total processing time
            person_profiles: Dict of track profiles with averaged embeddings (required for HDF5 export)
        """
        self.exporter.finalize_data(processing_time)
        
        # Determine the base output path
        base_path = self.output_path or output_path
        
        # Export HDF5 files if enabled
        if self.enable_hdf5 and person_profiles:
            # For HDF5, use base_path as directory (add suffix if it doesn't look like a dir)
            if not base_path.endswith('_hdf5'):
                hdf5_dir = base_path + '_hdf5'
            else:
                hdf5_dir = base_path
            self.exporter.export_tracks_to_hdf5(hdf5_dir, person_profiles)
        
        # Always export JSON
        # Ensure .json extension
        if base_path.endswith('_hdf5'):
            json_path = base_path.replace('_hdf5', '.json')
        elif not base_path.endswith('.json'):
            json_path = base_path + '.json'
        else:
            json_path = base_path
            
        self.exporter.export_to_json(json_path)