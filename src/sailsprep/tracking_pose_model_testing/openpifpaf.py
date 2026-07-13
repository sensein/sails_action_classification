import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openpifpaf
import PIL
import PIL.Image
import requests
import torch
from IPython.display import display

net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net_cpu.to(device)

openpifpaf.decoder.CifSeeds.threshold = 0.5
openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)
keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6) #linewidth can be changed here


input_folder = '/input'
output_folder = '/outputs/Openpifpaf'
os.makedirs(output_folder, exist_ok=True)

video_files = [f for f in os.listdir(input_folder) if f.endswith('.mkv')]
print(f"Found {len(video_files)} videos.")

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    out_path = os.path.join(output_folder, video_file.replace('.mkv', '_pose.mp4'))

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_im = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        im = np.asarray(pil_im)
        data = openpifpaf.datasets.PilImageList([pil_im])
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        predictions = []
        for images_batch, _, __ in loader:
            predictions = processor.batch(net, images_batch, device=device)[0]

        with openpifpaf.show.image_canvas(im) as ax:
            keypoint_painter.annotations(ax, predictions)
            ax.axis('off')
            ax.figure.tight_layout(pad=0)
            ax.figure.canvas.draw()
            rgba = np.asarray(ax.figure.canvas.buffer_rgba())
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            annotated_frame = cv2.resize(rgb, (w, h))
            plt.close(ax.figure)

        out.write(annotated_frame)

        # Commented display
        # if frame_idx % 200 == 0:
        #     print(f"Displaying frame {frame_idx}")
        #     cv2.imshow("frame", annotated_frame)
        #     cv2.waitKey(1)

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished: {video_file} → {out_path}")

print("Done processing all videos.")
