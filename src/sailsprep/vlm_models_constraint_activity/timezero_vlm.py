"""
TimeZero VLM Implementation
TimeZero is based on Qwen2.5-VL architecture, trained on ActivityNet
"""

from sailsprep.vlm_testing.base_vlm import BaseVLM
from typing import Optional
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import av
import numpy as np


class TimeZeroVLM(BaseVLM):
    """TimeZero-ActivityNet proper implementation (Qwen2.5-VL based)"""
    
    def __init__(self, model_config, device='cuda:0'):
        super().__init__(model_config, device)
        self.load_model()
    
    def load_model(self):
        """Load TimeZero model (Qwen2.5-VL architecture)"""
        print(f"Loading {self.config.name}")
        print(f"Model ID: {self.config.model_id}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load processor
        print("Loading Qwen2VL processor")
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            local_files_only=True
        )
        
        # Load model with AutoModelForVision2Seq (handles Qwen2.5-VL)
        print("Loading model with AutoModelForVision2Seq")
        dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            local_files_only=True,
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"✓ {self.config.name} loaded successfully")
        print(f"  Model class: {type(self.model).__name__}")
        print(f"  {self.get_gpu_stats()}")
    
    def read_video_frames(self, video_path: str) -> Optional[str]:
        """Return video path for Qwen2VL (handles video internally)"""
        # Qwen2VL can handle video paths directly via nframes parameter
        return f"file://{video_path}"
    
    def predict_activity(self, video_path: str, prompt: str) -> Optional[str]:
        """Predict activity from video"""
        try:
            # Qwen2VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "max_pixels": 360 * 420,
                            "fps": 1.0,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
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
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
            
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return self.parse_numeric_context(response)
            
        except Exception as e:
            print(f"Error in map_activity_to_context: {e}")
            import traceback
            traceback.print_exc()
            return None
