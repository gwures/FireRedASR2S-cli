from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

TEMP_DIR = BASE_DIR / "temp"
PRETRAINED_MODELS_DIR = BASE_DIR / "pretrained_models"

TEMP_DIR.mkdir(exist_ok=True)
PRETRAINED_MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATHS = {
    "vad": str(PRETRAINED_MODELS_DIR / "FireRedVAD" / "VAD"),
    "asr": str(PRETRAINED_MODELS_DIR / "FireRedASR2-AED"),
}

ASR_CONFIG = {
    "use_gpu": True,
    "use_half": True,
    "beam_size": 3,
    "nbest": 1,
    "decode_max_len": 0,
    "softmax_smoothing": 1.25,
    "aed_length_penalty": 0.6,
    "eos_penalty": 1.0,
    "return_timestamp": True,
}

VAD_CONFIG = {
    "use_gpu": True,
    "smooth_window_size": 5,
    "speech_threshold": 0.4,
    "min_speech_frame": 50,
    "max_speech_frame": 1600,
    "min_silence_frame": 20,
    "merge_silence_frame": 0,
    "extend_speech_frame": 0,
    "chunk_max_frame": 30000,
}

PUNCTUATION_CONFIG = {
    "enabled": True,
    "endpoint": "https://api.siliconflow.cn/v1/chat/completions",
    "api_key": "sk-l4esdghjjjj",
    "model": "THUDM/GLM-4-9B-0414",
    "prompt": "为下方无标点的长文本进行标准化标点补全。绝不改动原文任何字词、语序，仅补充标点；不要其他解释：\n{text}",
    "max_timeout": 600,
    "max_concurrent": 5,
}
