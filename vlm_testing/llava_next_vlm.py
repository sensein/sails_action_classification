"""
LLaVA-Next-Video Implementation
Supports: LLaVA-Next-Video-Qwen2, LLaVA-Next-Video-Llama3, LLaVA-Next-Video-Mistral
"""

import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from typing import Optional
from pathlib import Path
import av
import numpy as np

from base_vlm import BaseVLM


class LLaVANextVideoVLM(BaseVLM):
    """LLaVA-Next-Video model implementation"""
    
    def __init__(self, model_config, device='cuda:0'):
        super().__init__(model_config, device)
        self.load_model()
    
    def load_model(self):
        """Load LLaVA-Next-Video model and processor"""

        print(f"Loading {self.config.name}")
        print(f"Model ID: {self.config.model_id}")
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Load processor
        print("Loading processor...")
        self.processor = LlavaNextVideoProcessor.from_pretrained(
            self.config.model_id
        )
        
        # Load model
        print("Loading model...")
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="cuda:0",
        )
        
        self.model.eval()
        
        print("Model loaded successfully!")
        print(f"Max frames: {self.config.max_frames}")
        print(f"dtype: {self.config.dtype}")
    
    def _read_video_pyav(self, video_path: str, num_frames: int = 32):
        """Read video frames using PyAV (LLaVA-Next format)"""
        container = av.open(video_path)
        
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            # Estimate from duration and fps
            stream = container.streams.video[0]
            total_frames = int(stream.duration * stream.time_base * stream.average_rate)
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == num_frames:
                break
        
        container.close()
        return np.stack(frames)
    
    def predict_activity(self, video_path: str, prompt: str) -> Optional[str]:
        """Predict activity from video"""
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Read video
            video_frames = self._read_video_pyav(video_path, self.config.max_frames)
            
            # Prepare conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            # Process inputs
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, videos=video_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                output_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"Error in predict_activity for {video_path}: {e}")
            return None
    
    def map_activity_to_context(self, video_path: str, activity_description: str, prompt: str) -> Optional[int]:
        """Map activity to context category"""
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Read video
            video_frames = self._read_video_pyav(video_path, self.config.max_frames)
            
            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            # Process inputs
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, videos=video_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                output_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # Parse numeric response
            return self.parse_numeric_context(generated_text)
            
        except Exception as e:
            print(f"Error in map_activity_to_context for {video_path}: {e}")
            return None
