"""
SmolVLM2 VLM Implementation
Uses SmolVLMForConditionalGeneration
"""

from sailsprep.vlm_testing.base_vlm import BaseVLM
from typing import Optional
import torch
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
import av
import numpy as np
from PIL import Image


class SmolVLMVLM(BaseVLM):
    """SmolVLM2-500M proper implementation"""
    
    def __init__(self, model_config, device='cuda:0'):
        super().__init__(model_config, device)
        self.load_model()
    
    def load_model(self):
        """Load SmolVLM model"""
        print(f"Loading {self.config.name}")
        print(f"Model ID: {self.config.model_id}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Load model with correct class
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            local_files_only=True,
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"{self.config.name} loaded successfully")
        print(f"Model class: {type(self.model).__name__}")
        print(f"{self.get_gpu_stats()}")
    
    def read_video_frames(self, video_path: str) -> Optional[list]:
        """Read video frames as PIL Images"""
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            total_frames = stream.frames
            if total_frames == 0:
                total_frames = int(stream.duration * stream.time_base * stream.average_rate)
            
            indices = np.linspace(0, total_frames - 1, self.config.max_frames, dtype=int)
            frames = []
            
            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    img = Image.fromarray(frame.to_ndarray(format='rgb24'))
                    frames.append(img)
                if len(frames) >= self.config.max_frames:
                    break
            
            container.close()
            return frames if frames else None
            
        except Exception as e:
            print(f"Error reading video: {e}")
            return None
    
    def predict_activity(self, video_path: str, prompt: str) -> Optional[str]:
        """Predict activity from video"""
        try:
            frames = self.read_video_frames(video_path)
            if frames is None:
                return None
            
            # SmolVLM format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=text_prompt,
                images=frames,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return response
            
        except Exception as e:
            print(f"Error in predict_activity: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def map_activity_to_context(self, video_path: str, activity_description: str, prompt: str) -> Optional[int]:
        """Map activity to context"""
        try:
            frames = self.read_video_frames(video_path)
            if frames is None:
                return None
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text_prompt,
                images=frames,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return self.parse_numeric_context(response)
            
        except Exception as e:
            print(f"Error in map_activity_to_context: {e}")
            import traceback
            traceback.print_exc()
            return None
