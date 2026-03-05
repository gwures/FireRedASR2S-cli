# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang, Yan Jia, Junjie Chen, Wenpeng Li)

import logging
import re
from dataclasses import dataclass, field

import soundfile as sf
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

logger = logging.getLogger("fireredasr2s.asr_system")

_BLANK_SIL_SEARCH_PATTERN = re.compile(r"(<blank>)|(<sil>)")


@dataclass
class FireRedAsr2SystemConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    asr_config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)
    asr_batch_size: int = 1
    enable_vad: bool = True


class FireRedAsr2System:
    def __init__(self, config):
        c = config
        self.vad = FireRedVad.from_pretrained(c.vad_model_dir, c.vad_config) if c.enable_vad else None
        self.asr = FireRedAsr2.from_pretrained(c.asr_type, c.asr_model_dir, c.asr_config)
        self.config = config

    def process(self, wav_path, uttid="tmpid"):
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur = wav_np.shape[0]/sample_rate

        # 1. VAD
        if self.config.enable_vad:
            vad_result, prob = self.vad.detect(wav_path)
            vad_segments = vad_result["timestamps"]
            logger.info(f"VAD: {vad_result}")
        else:
            vad_segments = [(0, dur)]
            vad_result = {"timestamps" : vad_segments}

        # 2. VAD output to ASR input
        asr_results = []
        batch_asr_uttid = []
        batch_asr_wav = []
        for j, (start_s, end_s) in enumerate(vad_segments):
            wav_segment = wav_np[int(start_s*sample_rate):int(end_s*sample_rate)]
            vad_uttid = f"{uttid}_s{int(start_s*1000)}_e{int(end_s*1000)}"
            batch_asr_uttid.append(vad_uttid)
            batch_asr_wav.append((sample_rate, wav_segment))
            if len(batch_asr_uttid) < self.config.asr_batch_size and j != len(vad_segments) - 1:
                continue

            # 3. ASR
            batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
            logger.info(f"ASR: {batch_asr_results}")
            batch_asr_results = [a for a in batch_asr_results if not _BLANK_SIL_SEARCH_PATTERN.search(a["text"])]
            asr_results.extend(batch_asr_results)

            batch_asr_uttid = []
            batch_asr_wav = []

        # 4. Put all together & Format
        sentences = []
        words = []
        for asr_result in asr_results:
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            if self.config.asr_config.return_timestamp:
                sub_sentences = [{
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": asr_result["text"],
                    "asr_confidence": asr_result["confidence"],
                }]
                sentences.extend(sub_sentences)
            else:
                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": asr_result["text"],
                    "asr_confidence": asr_result["confidence"],
                }
                sentences.append(sentence)
            
            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    word = {"start_ms": int(s*1000+start_ms), "end_ms":int(e*1000+start_ms), "text": w}
                    words.append(word)

        vad_segments_ms = [(int(s*1000), int(e*1000)) for s, e in vad_result["timestamps"]]
        text = "".join(s["text"] for s in sentences)

        result = {
            "uttid": uttid,
            "text": text,
            "sentences": sentences,
            "vad_segments_ms": vad_segments_ms,
            "dur_s": dur,
            "words": words,
            "wav_path": wav_path
        }
        return result
