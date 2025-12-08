"""Model factory for VLM instances"""

from config import get_model_config
from base_vlm import BaseVLM


def create_vlm(model_name: str, device: str = 'cuda:0') -> BaseVLM:
    """Create VLM instance
    
    Args:
        model_name: Model name from config
        device: CUDA device
        
    Returns:
        VLM instance
    """
    config = get_model_config(model_name)
    
    if config.model_family == 'llava-next':
        from llava_next_vlm import LLaVANextVideoVLM
        return LLaVANextVideoVLM(config, device=device)
    
    elif config.model_family == 'qwen2-vl':
        from qwen2_vl_vlm import Qwen2VLVLM
        return Qwen2VLVLM(config, device=device)
    
    elif config.model_family == 'timezero':
        from timezero_vlm import TimeZeroVLM
        return TimeZeroVLM(config, device=device)
    
    elif config.model_family == 'smolvlm':
        from smolvlm_vlm import SmolVLMVLM
        return SmolVLMVLM(config, device=device)
    
    elif config.model_family == 'videollama2':
        from videollama2_vlm import VideoLLaMA2VLM
        return VideoLLaMA2VLM(config, device=device)
    
    else:
        raise ValueError(f"Unknown model family: {config.model_family}")
