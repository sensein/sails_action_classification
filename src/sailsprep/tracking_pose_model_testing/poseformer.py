import os
import shutil
import subprocess

input_folder = '/input'
output_folder = '/outputs/PoseFormer'
video_name = 'video.mkv'

os.makedirs(output_folder, exist_ok=True)

input_path = os.path.join(input_folder, video_name)
trimmed_video_path = f"/content/poseformer_demo/demo/video/{video_name}"

subprocess.run([
    'ffmpeg', '-y', '-i', input_path,
    '-t', '10', '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
    '-c:a', 'aac', '-b:a', '128k',
    trimmed_video_path
], check=True)



output_video_path = f"/poseformer_demo/demo/output/{video_name.split('.')[0]}/{video_name.split('.')[0]}.mp4"
if os.path.exists(output_video_path):
    shutil.copy(output_video_path, os.path.join(output_folder, f"{video_name.split('.')[0]}.mp4"))


