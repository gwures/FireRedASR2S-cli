# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang)

import logging
import os
from dataclasses import dataclass

import torch

from .core.audio_feat import AudioFeat
from .core.detect_model import DetectModel
from .core.vad_postprocessor import VadPostprocessor

logger = logging.getLogger(__name__)


@dataclass
class FireRedVadConfig:
    use_gpu: bool = False
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000  # 20s
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000  # 300s
    def __post_init__(self):
        if self.speech_threshold < 0 or self.speech_threshold > 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if self.min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")



class FireRedVad:
    @classmethod
    def from_pretrained(cls, model_dir, config=FireRedVadConfig()):
        # Build Feat Extractor
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        audio_feat = AudioFeat(cmvn_path)

        # Build Model
        vad_model = DetectModel.from_pretrained(model_dir)
        if config.use_gpu:
            vad_model.cuda()
        else:
            vad_model.cpu()

        # Build Postprocessor
        vad_postprocessor = VadPostprocessor(
            config.smooth_window_size,
            config.speech_threshold,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame,
            config.merge_silence_frame,
            config.extend_speech_frame)
        return cls(audio_feat, vad_model, vad_postprocessor, config)

    def __init__(self, audio_feat, vad_model, vad_postprocessor, config):
        self.audio_feat = audio_feat
        self.vad_model = vad_model
        self.vad_postprocessor = vad_postprocessor
        self.config = config

    def detect(self, audio, do_postprocess=True):
        feats, dur = self.audio_feat.extract(audio)
        
        if self.config.use_gpu:
            feats = feats.pin_memory()
            feats = feats.cuda(non_blocking=True)

        with torch.inference_mode():
            if feats.size(0) <= self.config.chunk_max_frame:
                probs, _ = self.vad_model.forward(feats.unsqueeze(0))
                probs = probs.cpu().squeeze()
            else:
                logger.warning(f"Too long input, split every {self.config.chunk_max_frame} frames")
                chunk_probs = []
                chunks = feats.split(self.config.chunk_max_frame, dim=0)
                for chunk in chunks:
                    chunk_prob, _ = self.vad_model.forward(chunk.unsqueeze(0))
                    chunk_probs.append(chunk_prob.cpu())
                probs = torch.cat(chunk_probs, dim=1)
                probs = probs.squeeze()

        if not do_postprocess:
            return None, probs

        decisions = self.vad_postprocessor.process(probs.tolist())
        starts_ends_s = self.vad_postprocessor.decision_to_segment(decisions, dur)

        result = {"dur": round(dur, 3),
                  "timestamps": starts_ends_s}
        if isinstance(audio, str):
            result["wav_path"] = audio
        return result, probs

    def detect_batch(self, audios, do_postprocess=True):
        """
        批量 VAD 推理
        
        Args:
            audios: 音频列表，每个元素格式为 (sample_rate, wav_np) 或文件路径
            do_postprocess: 是否进行后处理
            
        Returns:
            results: 结果列表，每个元素为 {"dur": float, "timestamps": List}
            probs_list: 概率列表
        """
        feats_list = []
        durs = []
        
        for audio in audios:
            feats, dur = self.audio_feat.extract(audio)
            feats_list.append(feats)
            durs.append(dur)
        
        if len(feats_list) == 0:
            return [], []
        
        max_len = max(f.size(0) for f in feats_list)
        batch_size = len(feats_list)
        
        padded_feats = torch.zeros(batch_size, max_len, feats_list[0].size(1))
        feat_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, feats in enumerate(feats_list):
            feat_len = feats.size(0)
            padded_feats[i, :feat_len, :] = feats
            feat_lengths[i] = feat_len
        
        if self.config.use_gpu:
            padded_feats = padded_feats.pin_memory()
            padded_feats = padded_feats.cuda(non_blocking=True)
            feat_lengths = feat_lengths.cuda()
        
        with torch.inference_mode():
            probs, _ = self.vad_model.forward(padded_feats, input_lengths=feat_lengths)
            probs = probs.cpu()
        
        results = []
        probs_list = []
        
        for i, (dur, feat_len) in enumerate(zip(durs, feat_lengths)):
            prob = probs[i, :feat_len].squeeze()
            
            if not do_postprocess:
                results.append(None)
                probs_list.append(prob)
                continue
            
            decisions = self.vad_postprocessor.process(prob.tolist())
            starts_ends_s = self.vad_postprocessor.decision_to_segment(decisions, dur)
            
            result = {"dur": round(dur, 3), "timestamps": starts_ends_s}
            if isinstance(audios[i], str):
                result["wav_path"] = audios[i]
            results.append(result)
            probs_list.append(prob)
        
        return results, probs_list
