#!/usr/bin/env python3
"""
OpenPose Batch Video Processing Script
Processes multiple videos from CSV with pose, face, and hand estimation
Uses FFmpeg for H.264 encoding
"""
import cv2
import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
from openpose import pyopenpose as op

def check_ffmpeg_available():
    """Check if ffmpeg is available in the system"""
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def process_video(input_video, output_video, model_folder="/opt/openpose/models/", use_ffmpeg=None):
    """
    Process video with OpenPose (body + face + hands) and save results
    
    Args:
        input_video: Path to input video file
        output_video: Path to output video file
        model_folder: Path to OpenPose models directory
        use_ffmpeg: If True, use FFmpeg; if False, use OpenCV; if None, auto-detect
    """
    
    # Auto-detect ffmpeg availability
    if use_ffmpeg is None:
        use_ffmpeg = check_ffmpeg_available()
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found!")
        return False
    
    # Configure OpenPose with face and hand detection
    params = dict()
    params["model_folder"] = model_folder
    params["net_resolution"] = "-1x368"  # Adjust for speed/accuracy trade-off
    params["face"] = True  # Enable face keypoint detection (70 keypoints)
    params["hand"] = True  # Enable hand keypoint detection (21 keypoints per hand)
    params["model_pose"] = "BODY_25"  # Use BODY_25 model (25 keypoints)
    
    # Initialize OpenPose
    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
    except Exception as e:
        print(f"Error initializing OpenPose: {e}")
        return False
    
    # Open input video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video '{input_video}'")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Encoder: {'FFmpeg (H.264)' if use_ffmpeg else 'OpenCV (X264)'}")
    
    # Setup video writer based on availability
    if use_ffmpeg:
        # Setup ffmpeg for H.264 encoding - using RGB format
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", 
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", 
            "-r", str(fps), 
            "-i", "-",
            "-an",
            "-c:v", "libx264", 
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
            output_video
        ]
        
        # Start ffmpeg process
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"  Error starting FFmpeg: {e}")
            cap.release()
            return False
    else:
        # Use OpenCV VideoWriter with X264
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"  Warning: X264 failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"  Error: Cannot create video writer!")
                cap.release()
                return False
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame with OpenPose
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            
            # Get output frame with pose overlay (should include body, face, and hands)
            output_frame = datum.cvOutputData
            
            # Write frame based on encoder type
            if use_ffmpeg:
                # Convert BGR to RGB for ffmpeg
                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                # Write frame to ffmpeg stdin
                try:
                    ffmpeg_process.stdin.write(rgb_frame.tobytes())
                except BrokenPipeError:
                    print(f"  Error: FFmpeg pipe broken")
                    break
            else:
                # Write frame directly with OpenCV
                out.write(output_frame)
            
            frame_count += 1
            
            # Print progress every 30 frames
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Finalize based on encoder type
        if use_ffmpeg:
            # Close ffmpeg stdin and wait for process to finish
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            
            if ffmpeg_process.returncode == 0:
                print(f"  ✓ Processing complete!")
            else:
                print(f"  ✗ FFmpeg encoding failed with code {ffmpeg_process.returncode}")
                return False
        else:
            # Release OpenCV writer
            out.release()
            print(f"  ✓ Processing complete!")
        
    except KeyboardInterrupt:
        print("\n  Processing interrupted by user")
        if use_ffmpeg:
            ffmpeg_process.terminate()
        else:
            out.release()
        return False
    except Exception as e:
        print(f"  Error during processing: {e}")
        if use_ffmpeg:
            ffmpeg_process.terminate()
        else:
            out.release()
        return False
    finally:
        # Clean up
        cap.release()
        if use_ffmpeg and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
    
    return True

def process_csv_videos(csv_path, output_dir, model_folder="/opt/openpose/models/"):
    """
    Process all videos listed in CSV file
    
    Args:
        csv_path: Path to CSV file with 'BidsProcessed' column
        output_dir: Directory to save output videos
        model_folder: Path to OpenPose models directory
    """
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found!")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Check if BidsProcessed column exists
    if 'BidsProcessed' not in df.columns:
        print(f"Error: 'BidsProcessed' column not found in CSV!")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    # Get list of video files
    video_paths = df['BidsProcessed'].dropna().tolist()
    total_videos = len(video_paths)
    
    print(f"\nFound {total_videos} videos to process")
    print("=" * 70)
    
    # Track processing statistics
    successful = 0
    failed = 0
    
    # Process each video
    for idx, input_video in enumerate(video_paths, 1):
        print(f"\n[{idx}/{total_videos}] Processing: {input_video}")
        
        # Create output filename
        input_path = Path(input_video)
        output_filename = f"{input_path.stem}_openpose{input_path.suffix}"
        output_video = os.path.join(output_dir, output_filename)
        
        # Skip if output already exists
        if os.path.exists(output_video):
            print(f"  ⚠ Output already exists, skipping: {output_filename}")
            continue
        
        # Process video
        success = process_video(input_video, output_video, model_folder)
        
        if success:
            successful += 1
            print(f"  ✓ Saved to: {output_filename}")
        else:
            failed += 1
            print(f"  ✗ Failed to process video")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total videos: {total_videos}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already exists): {total_videos - successful - failed}")

if __name__ == "__main__":
    # Default paths
    csv_path = "/home/aparnabg/orcd/scratch/csv2_filtered_multiple_people.csv"
    output_dir = "/home/aparnabg/orcd/scratch/openpose_output"
    model_folder = "/home/aparnabg/orcd/scratch/openpose/models"
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        model_folder = sys.argv[3]
    
    # Check encoder availability
    has_ffmpeg = check_ffmpeg_available()
    encoder_info = "FFmpeg libx264 (H.264)" if has_ffmpeg else "OpenCV X264/mp4v (fallback)"
    
    print("OpenPose Batch Video Processor")
    print("=" * 70)
    print(f"CSV File: {csv_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Model Folder: {model_folder}")
    print(f"Models: BODY_25 (25 keypoints) + FACE (70 keypoints) + HANDS (21 per hand)")
    print(f"Encoder: {encoder_info}")
    if has_ffmpeg:
        print(f"Settings: CRF=23, preset=medium")
    print("=" * 70)
    
    process_csv_videos(csv_path, output_dir, model_folder)