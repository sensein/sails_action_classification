"""
Qwen2-VL Implementation
Supports: Qwen2-VL-7B-Instruct (video variant)
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Optional
import av
import numpy as np
from qwen_vl_utils import process_vision_info

from base_vlm import BaseVLM


class Qwen2VLVLM(BaseVLM):
    """Qwen2-VL model implementation"""
    
    def __init__(self, model_config, device='cuda:0'):
        super().__init__(model_config, device)
        self.load_model()
    
    def load_model(self):
        """Load Qwen2-VL model and processor"""
        print(f"Loading {self.config.name}")
        print(f"Model ID: {self.config.model_id}")

        
        # GPU info
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Load model
        print("Loading Qwen2-VL model")
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="cuda:0"
        )
        
        # Load processor
        print("Loading processor")
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        
        self.model.eval()
        
        print(" Model loaded successfully!")
        print(f"Max frames: {self.config.max_frames}")
        print(f"dtype: {self.config.dtype}")
    
    def _prepare_video_messages(self, video_path: str, prompt: str):
        """Prepare messages in Qwen2-VL format"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages
    
    def predict_activity(self, video_path: str, prompt: str) -> Optional[str]:
        """Predict activity from video"""
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Prepare messages
            messages = self._prepare_video_messages(video_path, prompt)
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Trim and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            print(f"Error in predict_activity for {video_path}: {e}")
            return None
    
    def map_activity_to_context(self, video_path: str, activity_description: str, prompt: str) -> Optional[int]:
        """Map activity to context category"""
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Prepare messages
            messages = self._prepare_video_messages(video_path, prompt)
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            # Trim and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse numeric response
            return self.parse_numeric_context(output_text)
            
        except Exception as e:
            print(f"Error in map_activity_to_context for {video_path}: {e}")
            return None
