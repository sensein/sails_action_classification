import json
import locale
import os
from argparse import ArgumentParser
from base64 import b64encode
from os import path

import cv2
import numpy as np
import torch
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame
from deva.ext.with_text_processor import process_frame_with_text as process_frame_text
from deva.inference.demo_utils import flush_buffer
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.model.network import DEVA
from IPython.display import HTML
from tqdm import tqdm

locale.getpreferredencoding = lambda: "UTF-8"


try:
  import groundingdino
  from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
  import GroundingDINO
  from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel



torch.autograd.set_grad_enabled(False)

# for id2rgb
np.random.seed(42)

# default parameters
parser = ArgumentParser()
add_common_eval_args(parser)
add_ext_eval_args(parser)
add_text_default_args(parser)

# load model and config
args = parser.parse_args([])
cfg = vars(args)
cfg['enable_long_term'] = True

# Load our checkpoint
deva_model = DEVA(cfg).cuda().eval()
if args.model is not None:
    model_weights = torch.load(args.model)
    deva_model.load_weights(model_weights)
else:
    print('No model loaded.')

gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')




cfg['enable_long_term_count_usage'] = True
cfg['max_num_objects'] = 50
cfg['size'] = 480
cfg['DINO_THRESHOLD'] = 0.35
cfg['amp'] = True
cfg['chunk_size'] = 4
cfg['detection_every'] = 5
cfg['max_missed_detection_count'] = 10
cfg['sam_variant'] = 'original'
cfg['temporal_setting'] = 'online' # semionline usually works better; but online is faster for this demo
cfg['pluralize'] = True


SOURCE_VIDEO_DIR = "video_folder"
OUTPUT_VIDEO_DIR = "output"
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)


CLASSES = ['person']
cfg['prompt'] = '.'.join(CLASSES)
cfg['DINO_THRESHOLD'] = 0.5


deva = DEVAInferenceCore(deva_model, config=cfg)
deva.next_voting_frame = cfg['num_voting_frames'] - 1
deva.enabled_long_id()


video_files = [f for f in os.listdir(SOURCE_VIDEO_DIR) if f.endswith(('.mp4', '.webm', '.avi'))]

for video_file in video_files:
    source_path = os.path.join(SOURCE_VIDEO_DIR, video_file)
    output_filename = os.path.splitext(video_file)[0] + '_segmented.webm'
    output_path = os.path.join(OUTPUT_VIDEO_DIR, output_filename)

    print(f"\nProcessing: {video_file}")
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ti = 0
    writer_initialized = False

    result_saver = ResultSaver(None, None, dataset='gradio', object_manager=deva.object_manager)

    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=video_file) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if not writer_initialized:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (w, h))
                    writer_initialized = True
                    result_saver.writer = writer

                process_frame_text(deva, gd_model, sam_model, 'null.png', result_saver, ti, image_np=frame)
                ti += 1
                pbar.update(1)

        flush_buffer(deva, result_saver)
        if writer_initialized:
            writer.release()
    cap.release()
    deva.clear_buffer()

print("All videos processed and saved in:", OUTPUT_VIDEO_DIR)
