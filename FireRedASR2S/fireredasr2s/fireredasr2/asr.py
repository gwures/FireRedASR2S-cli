# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import gc
import logging
import os
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .data.asr_feat import ASRFeatExtractor
from .models.fireredasr_aed import FireRedAsrAed
from .models.fireredasr_llm import FireRedAsrLlm
from .models.lstm_lm import LstmLm
from .models.param import count_model_parameters
from .tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from .tokenizer.llm_tokenizer import LlmTokenizerWrapper

logger = logging.getLogger(__name__)

_BLANK_SIL_PATTERN = re.compile(r"(<blank>)|(<sil>)")


@dataclass
class FireRedAsr2Config:
    use_gpu: bool = True
    use_half: bool = False
    beam_size: int = 3
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1.25
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    return_timestamp: bool = False
    decode_min_len: int = 0
    repetition_penalty: float = 1.0
    llm_length_penalty: float = 0.0
    temperature: float = 1.0
    elm_dir: str = ""
    elm_weight: float = 0.0

    def __post_init__(self):
        pass


class FireRedAsr2:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir, config=FireRedAsr2Config()):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path = os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir
            )
        elm = None
        if config.elm_dir:
            assert os.path.exists(config.elm_dir), f"{config.elm_dir}"
            model_path = os.path.join(config.elm_dir, "model.pth.tar")
            elm = load_lstm_lm(model_path)
            elm.eval()
            logger.info(elm)
        count_model_parameters(model)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer, elm, config)

    def __init__(self, asr_type, feat_extractor, model, tokenizer, elm, config):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer
        self.elm = elm
        self.config = config
        logger.info(self.config)
        if self.config.use_gpu:
            if self.config.use_half:
                self.model.half()
            self.model.cuda()
            if self.elm:
                self.elm.cuda()
        else:
            self.model.cpu()

    @contextmanager
    def _inference_session(self):
        try:
            yield
        finally:
            gc.collect()
            if self.config.use_gpu and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def release_resources(self):
        if hasattr(self, "model"):
            del self.model
            self.model = None
        if hasattr(self, "feat_extractor"):
            del self.feat_extractor
            self.feat_extractor = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, "elm"):
            del self.elm
            self.elm = None

        gc.collect()
        if self.config.use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        logger.info("FireRedAsr2 resources released")

    def transcribe(self, batch_uttid, batch_wav_path):
        with torch.inference_mode():
            batch_uttid_origin = batch_uttid
            try:
                feats, lengths, durs, batch_wav_path, batch_uttid = self.feat_extractor(
                    batch_wav_path, batch_uttid
                )
                if feats is None:
                    return [
                        {"uttid": uttid, "text": ""} for uttid in batch_uttid_origin
                    ]
            except Exception as e:
                logger.error(f"Failed to extract features: {e}", exc_info=True)
                return [{"uttid": uttid, "text": ""} for uttid in batch_uttid_origin]
            total_dur = sum(durs)
            if self.config.use_gpu:
                dtype = torch.float16 if self.config.use_half else torch.float32
                feats = feats.to(device="cuda", dtype=dtype, non_blocking=True)
                lengths = lengths.to(device="cuda", non_blocking=True)

            if self.asr_type == "aed":
                start_time = time.time()

                hyps = []
                try:
                    hyps = self.model.transcribe(
                        feats,
                        lengths,
                        self.config.beam_size,
                        self.config.nbest,
                        self.config.decode_max_len,
                        self.config.softmax_smoothing,
                        self.config.aed_length_penalty,
                        self.config.eos_penalty,
                        self.config.return_timestamp,
                        self.elm,
                        self.config.elm_weight,
                    )
                except Exception:
                    traceback.print_exc()
                    hyps = []
                finally:
                    del feats, lengths

                elapsed = time.time() - start_time
                rtf = elapsed / total_dur if total_dur > 0 else 0

                results = []
                for uttid, wav, hyp, dur in zip(
                    batch_uttid, batch_wav_path, hyps, durs
                ):
                    hyp = hyp[0]
                    hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                    text = self.tokenizer.detokenize(hyp_ids)
                    text = _BLANK_SIL_PATTERN.sub("", text)
                    results.append(
                        {
                            "uttid": uttid,
                            "text": text.lower(),
                            "confidence": round(hyp["confidence"].cpu().item(), 3),
                            "dur_s": round(dur, 3),
                            "rtf": f"{rtf:.4f}",
                        }
                    )
                    if isinstance(wav, str):
                        results[-1]["wav"] = wav
                    if self.config.return_timestamp:
                        results[-1]["timestamp"] = self._get_and_fix_timestamp(
                            hyp, hyp_ids, dur
                        )
                    del hyp_ids

                del hyps
                return results

            elif self.asr_type == "llm":
                input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""] * feats.size(0),
                    tokenizer=self.tokenizer,
                    max_len=128,
                    decode=True,
                )
                if self.config.use_gpu:
                    input_ids = input_ids.to(device="cuda", non_blocking=True)
                    attention_mask = attention_mask.to(device="cuda", non_blocking=True)
                start_time = time.time()

                texts = []
                try:
                    generated_ids = self.model.transcribe(
                        feats,
                        lengths,
                        input_ids,
                        attention_mask,
                        self.config.beam_size,
                        self.config.decode_max_len,
                        self.config.decode_min_len,
                        self.config.repetition_penalty,
                        self.config.llm_length_penalty,
                        self.config.temperature,
                    )
                    texts = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                except Exception:
                    texts = []
                finally:
                    del feats, lengths, input_ids, attention_mask
                    if "generated_ids" in locals():
                        del generated_ids
                elapsed = time.time() - start_time
                rtf = elapsed / total_dur if total_dur > 0 else 0
                results = []
                for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                    results.append(
                        {"uttid": uttid, "text": text.lower(), "rtf": f"{rtf:.4f}"}
                    )
                    if isinstance(wav, str):
                        results[-1]["wav"] = wav
                return results

    def transcribe_from_cached_fbank(self, cached_fbank, segment_infos):
        """
        从缓存的 fbank 特征进行转录，避免重复计算特征

        Args:
            cached_fbank: VAD 阶段缓存的原始 fbank tensor, shape [T, 80]
            segment_infos: List of (start_frame, end_frame, uttid, dur) tuples

        Returns:
            List of result dicts
        """
        with torch.inference_mode():
            try:
                feats, lengths, durs, batch_uttid = (
                    self.feat_extractor.extract_from_cached_fbank(
                        cached_fbank, segment_infos
                    )
                )
                if feats is None:
                    return [{"uttid": seg[2], "text": ""} for seg in segment_infos]
            except Exception as e:
                logger.error(
                    f"Failed to extract features from cache: {e}", exc_info=True
                )
                return [{"uttid": seg[2], "text": ""} for seg in segment_infos]

            total_dur = sum(durs)
            if self.config.use_gpu:
                dtype = torch.float16 if self.config.use_half else torch.float32
                feats = feats.to(device="cuda", dtype=dtype, non_blocking=True)
                lengths = lengths.to(device="cuda", non_blocking=True)

            if self.asr_type == "aed":
                start_time = time.time()

                hyps = []
                try:
                    hyps = self.model.transcribe(
                        feats,
                        lengths,
                        self.config.beam_size,
                        self.config.nbest,
                        self.config.decode_max_len,
                        self.config.softmax_smoothing,
                        self.config.aed_length_penalty,
                        self.config.eos_penalty,
                        self.config.return_timestamp,
                        self.elm,
                        self.config.elm_weight,
                    )
                except Exception:
                    traceback.print_exc()
                    hyps = []
                finally:
                    del feats, lengths

                elapsed = time.time() - start_time
                rtf = elapsed / total_dur if total_dur > 0 else 0

                results = []
                for uttid, hyp, dur in zip(batch_uttid, hyps, durs):
                    hyp = hyp[0]
                    hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                    text = self.tokenizer.detokenize(hyp_ids)
                    text = _BLANK_SIL_PATTERN.sub("", text)
                    results.append(
                        {
                            "uttid": uttid,
                            "text": text.lower(),
                            "confidence": round(hyp["confidence"].cpu().item(), 3),
                            "dur_s": round(dur, 3),
                            "rtf": f"{rtf:.4f}",
                        }
                    )
                    if self.config.return_timestamp:
                        results[-1]["timestamp"] = self._get_and_fix_timestamp(
                            hyp, hyp_ids, dur
                        )
                    del hyp_ids

                del hyps
                return results

            elif self.asr_type == "llm":
                input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""] * feats.size(0),
                    tokenizer=self.tokenizer,
                    max_len=128,
                    decode=True,
                )
                if self.config.use_gpu:
                    input_ids = input_ids.to(device="cuda", non_blocking=True)
                    attention_mask = attention_mask.to(device="cuda", non_blocking=True)
                start_time = time.time()

                texts = []
                try:
                    generated_ids = self.model.transcribe(
                        feats,
                        lengths,
                        input_ids,
                        attention_mask,
                        self.config.beam_size,
                        self.config.decode_max_len,
                        self.config.decode_min_len,
                        self.config.repetition_penalty,
                        self.config.llm_length_penalty,
                        self.config.temperature,
                    )
                    texts = self.tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                except Exception:
                    texts = []
                finally:
                    del feats, lengths, input_ids, attention_mask
                    if "generated_ids" in locals():
                        del generated_ids
                elapsed = time.time() - start_time
                rtf = elapsed / total_dur if total_dur > 0 else 0
                results = []
                for uttid, text in zip(batch_uttid, texts):
                    results.append(
                        {"uttid": uttid, "text": text.lower(), "rtf": f"{rtf:.4f}"}
                    )
                return results

    def _get_and_fix_timestamp(self, hyp, hyp_ids, dur):
        def r3(x):
            return round(x, 3)

        if "timestamp" not in hyp or hyp["timestamp"] is None:
            timestamp = []
            avg_dur = dur / len(hyp_ids) if len(hyp_ids) > 0 else 0
            last_end = dur
            for i, hyp_id in enumerate(hyp_ids):
                token = self.tokenizer.detokenize([hyp_id], "", False)
                start = min(max(0, i * avg_dur), last_end)
                end = min((i + 1) * avg_dur, dur)
                last_end = end
                timestamp.append([token.lower(), r3(start), r3(end)])
        else:
            starts, ends = hyp["timestamp"]
            timestamp = []
            last_end = dur
            SHIFT = 0.06  # shift 40ms
            for hyp_id, start, end in zip(hyp_ids, starts, ends):
                token = self.tokenizer.detokenize([hyp_id], "", False)
                start = min(max(0, start - SHIFT), last_end)
                end = min(max(0, end - SHIFT), dur)
                last_end = end
                timestamp.append([token.lower(), r3(start), r3(end)])
        # Fix case: start == dur and end == dur
        for i in range(len(timestamp)):
            idx = -(i + 1)
            _, start, end = timestamp[idx]
            if abs(dur - start) < 0.001:
                logger.info(f"start before {timestamp[idx]}")
                timestamp[idx][1] = dur - (i + 1) * 0.001
                logger.info(f"start after {timestamp[idx]}")
            if i != 0 and abs(dur - end) < 0.001:
                logger.info(f"end before {timestamp[idx]}")
                timestamp[idx][2] = dur - i * 0.001
                logger.info(f"end after {timestamp[idx]}")
        timestamp = self.tokenizer.merge_spm_timestamp(timestamp)
        return timestamp


def load_fireredasr_aed_model(model_path):
    package = torch.load(
        model_path, map_location=lambda storage, loc: storage, weights_only=False
    )
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(
        model_path, map_location=lambda storage, loc: storage, weights_only=False
    )
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer


def load_lstm_lm(model_path):
    package = torch.load(
        model_path, map_location=lambda storage, loc: storage, weights_only=False
    )
    model = LstmLm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model
