"""Model configurations for VLMs
Todo: Add the all models tested"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    model_id: str
    model_family: str
    max_frames: int
    dtype: str
    supports_audio: bool = False


MODEL_REGISTRY = {
    'llava-next-7b': ModelConfig(
        name='llava-next-7b',
        model_id='llava-hf/LLaVA-NeXT-Video-7B-hf',
        model_family='llava-next',
        max_frames=32,
        dtype='bfloat16'
    ),
    'qwen2-vl-7b': ModelConfig(
        name='qwen2-vl-7b',
        model_id='Qwen/Qwen2.5-VL-7B-Instruct',
        model_family='qwen2-vl',
        max_frames=32,
        dtype='bfloat16'
    ),
    'timezero-7b': ModelConfig(
        name='timezero-7b',
        model_id='wwwyyy/TimeZero-ActivityNet-7B',
        model_family='timezero',
        max_frames=32,
        dtype='bfloat16'
    ),
    'smolvlm2-500m': ModelConfig(
        name='smolvlm2-500m',
        model_id='HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
        model_family='smolvlm',
        max_frames=32,
        dtype='bfloat16'
    ),
    'videollama2-7b': ModelConfig(
        name='videollama2-7b',
        model_id='DAMO-NLP-SG/VideoLLaMA2-7B',
        model_family='videollama2',
        max_frames=16,
        dtype='bfloat16'
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def list_models():
    return list(MODEL_REGISTRY.keys())
