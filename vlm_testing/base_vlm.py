"""
VLM Base Class
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict
from pathlib import Path
import torch


class BaseVLM(ABC):
    """
    Abstract base class for video-language models.
    All model implementations must inherit from this and implement required methods.
    """
    
    def __init__(self, model_config, device='cuda:0'):
        """
        Initialize the VLM model
        
        Args:
            model_config: ModelConfig object from vlm_config.py
            device: Device to run model on
        """
        self.config = model_config
        self.device = device
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Context mapping (standardized across all models)
        # 8 categories total
        self.context_map = {
            'special occasion': 1,
            'general social communication interaction': 2,
            'general social interaction': 2,  # Maps to same (typo in CSV)
            'motor play': 3,
            'daily routine': 4,
            'toy play': 5,
            'social routine': 6,
            'other': 7,
            'book share': 8,
        }
        
        self.numeric_to_context = {
            1: 'special occasion',
            2: 'general social communication interaction',
            3: 'motor play',
            4: 'daily routine',
            5: 'toy play',
            6: 'social routine',
            7: 'other',
            8: 'book share'
        }
        
    @abstractmethod
    def load_model(self):
        """Load the model, processor, and tokenizer"""
        pass
    
    @abstractmethod
    def predict_activity(self, video_path: str, prompt: str) -> Optional[str]:
        """
        Predict activity description from video
        
        Args:
            video_path: Path to video file
            prompt: Prompt for activity prediction
            
        Returns:
            Activity description string or None if error
        """
        pass
    
    @abstractmethod
    def map_activity_to_context(self, video_path: str, activity_description: str, prompt: str) -> Optional[int]:
        """
        Map activity description to context category
        
        Args:
            video_path: Path to video file
            activity_description: Predicted activity description
            prompt: Prompt for context mapping
            
        Returns:
            Context category number (1-8) or None if error
        """
        pass
    
    def process_video(self, video_path: str, activity_prompt: str, context_prompt: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Complete pipeline: predict activity and map to context
        
        Args:
            video_path: Path to video file
            activity_prompt: Prompt for activity prediction
            context_prompt: Prompt for context mapping (with {activity_description} placeholder)
            
        Returns:
            Tuple of (activity_description, context_number)
        """
        # Stage 1: Predict activity
        activity = self.predict_activity(video_path, activity_prompt)
        if activity is None:
            return None, None
        
        # Stage 2: Map to context
        context_prompt_filled = context_prompt.format(activity_description=activity)
        context = self.map_activity_to_context(video_path, activity, context_prompt_filled)
        
        return activity, context
    
    def get_gpu_stats(self) -> str:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        return "GPU not available"
    
    def cleanup(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def parse_numeric_context(response: str) -> Optional[int]:
        """
        Extract numeric context from model response
        Handles various response formats
        """
        import re
        response = str(response).strip().lower()
        
        # Try to extract numeric value first
        numeric_match = re.search(r'\b([1-8])\b', response)
        if numeric_match:
            return int(numeric_match.group(1))
        
        # Try to convert directly to int
        try:
            num = int(response)
            if 1 <= num <= 8:
                return num
        except (ValueError, TypeError):
            pass
        
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'name': self.config.name,
            'model_id': self.config.model_id,
            'family': self.config.model_family,
            'max_frames': self.config.max_frames,
            'supports_audio': self.config.supports_audio,
            'dtype': self.config.dtype,
            'device': self.device
        }
