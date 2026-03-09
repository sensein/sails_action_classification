"""VideoLLaMA2 VLM Implementation"""

from base_vlm import BaseVLM
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModel
import av
import numpy as np
from PIL import Image


class VideoLLaMA2VLM(BaseVLM):
    """VideoLLaMA2 implementation"""
    
    def __init__(self, model_config, device='cuda:0'):
        super().__init__(model_config, device)
        self.load_model()
    
    def load_model(self):
        """Load VideoLLaMA2 from HuggingFace"""
        print(f"Loading {self.config.name}")
        print(f"Model ID: {self.config.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            local_files_only=True,
            trust_remote_code=True
        )
        
        dtype = torch.float16 if self.config.dtype == "float16" else torch.bfloat16
        
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            local_files_only=True,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        print(f"Loaded {self.config.name}")
        print(f"Model class: {type(self.model).__name__}")
    
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
            
            text_input = f"USER: <video>\n{prompt} ASSISTANT:"
            
            text_tensor = self.tokenizer(
                text_input,
                return_tensors="pt"
            ).to(self.device)
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            video_tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=text_tensor['input_ids'],
                    pixel_values=video_tensor,
                    max_new_tokens=100,
                    do_sample=False,
                    num_beams=1
                )
            
            response = self.tokenizer.batch_decode(
                output_ids[:, text_tensor['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0].strip()
            
            return response
            
        except Exception as e:
            print(f"Error in predict_activity: {e}")
            return None
    
    def map_activity_to_context(self, video_path: str, activity_description: str, prompt: str) -> Optional[int]:
        """Map activity to context"""
        try:
            frames = self.read_video_frames(video_path)
            if frames is None:
                return None
            
            text_input = f"USER: <video>\n{prompt} ASSISTANT:"
            
            text_tensor = self.tokenizer(
                text_input,
                return_tensors="pt"
            ).to(self.device)
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            video_tensor = torch.stack([transform(frame) for frame in frames]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=text_tensor['input_ids'],
                    pixel_values=video_tensor,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1
                )
            
            response = self.tokenizer.batch_decode(
                output_ids[:, text_tensor['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0].strip()
            
            return self.parse_numeric_context(response)
            
        except Exception as e:
            print(f"Error in map_activity_to_context: {e}")
            return None
