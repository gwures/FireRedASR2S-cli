import gc
import heapq
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

_BLANK_SIL_PATTERN = re.compile(r"(<blank>)|(<sil>)")


def filter_timestamp_by_duration(timestamp: List, audio_dur: float) -> List:
    if not timestamp:
        return timestamp

    filtered_timestamp = []
    for item in timestamp:
        if len(item) != 3:
            filtered_timestamp.append(item)
            continue
        w, s, e = item
        s_time = float(s)
        e_time = float(e)

        if e_time <= audio_dur:
            filtered_timestamp.append((w, s_time, e_time))
        elif s_time < audio_dur:
            filtered_timestamp.append((w, s_time, audio_dur))

    return filtered_timestamp


@dataclass
class VadSegment:
    seg_idx: int
    start_s: float
    end_s: float
    uttid: str


@dataclass
class AudioProcessResult:
    audio_idx: int
    audio_path: str
    success: bool
    result: Optional[Dict] = None
    error_msg: Optional[str] = None
    processing_time_s: float = 0.0
    rtf: float = 0.0


logger = logging.getLogger("fireredasr2s.worker")

_asr_system_instance: Optional["MixedASRSystem"] = None
_asr_system_lock = threading.Lock()
_asr_system_config_hash: Optional[str] = None


def _compute_config_hash(config: Dict[str, Any]) -> str:
    import hashlib
    import json

    hashable_keys = ["model_paths", "asr", "vad", "punc", "max_batch_dur_s", "use_bfd"]
    hashable_config = {k: v for k, v in config.items() if k in hashable_keys}

    config_str = json.dumps(hashable_config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def get_or_create_asr_system(config: Dict[str, Any]) -> "MixedASRSystem":
    global _asr_system_instance, _asr_system_config_hash

    config_hash = _compute_config_hash(config)

    with _asr_system_lock:
        if _asr_system_instance is not None and _asr_system_config_hash == config_hash:
            logger.info("Reusing existing MixedASRSystem instance")
            return _asr_system_instance

        if _asr_system_instance is not None:
            logger.warning("Config changed, releasing old MixedASRSystem instance")
            if hasattr(_asr_system_instance, "release_resources"):
                _asr_system_instance.release_resources()
            _asr_system_instance = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        logger.info("Creating new MixedASRSystem instance")
        _asr_system_instance = MixedASRSystem(config)
        _asr_system_config_hash = config_hash
        return _asr_system_instance


def release_asr_system():
    global _asr_system_instance, _asr_system_config_hash

    with _asr_system_lock:
        if _asr_system_instance is not None:
            logger.info("Releasing MixedASRSystem instance")

            if hasattr(_asr_system_instance, "release_resources"):
                _asr_system_instance.release_resources()

            _asr_system_instance = None
            _asr_system_config_hash = None

            for _ in range(3):
                gc.collect()

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class MixedASRSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_paths = config["model_paths"]

        self.max_batch_dur_s = config.get("max_batch_dur_s", 64)
        self.use_bfd = config.get("use_bfd", False)

        asr_config = config.get("asr", {})
        use_fp16 = asr_config.get("use_half", True)
        return_timestamp = asr_config.get("return_timestamp", True)

        logger.info(
            f"MixedASRSystem created (max_batch_dur_s={self.max_batch_dur_s}, "
            f"use_bfd={self.use_bfd}, fp16={use_fp16}, timestamp={return_timestamp})"
        )

        self._load_vad()
        self._load_asr()
        self._load_punc()

        logger.info("VAD, ASR and Punc preloaded!")

    def _load_vad(self):
        logger.info("Loading VAD model...")
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

        vad_config = FireRedVadConfig(**self.config["vad"])
        self.vad = FireRedVad.from_pretrained(self.model_paths["vad"], vad_config)
        logger.info("VAD model loaded!")

    def _load_asr(self):
        logger.info("Loading ASR model...")
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

        asr_config = FireRedAsr2Config(**self.config["asr"])
        self.asr = FireRedAsr2.from_pretrained(
            "aed", self.model_paths["asr"], asr_config
        )
        logger.info("ASR model loaded!")

    def _load_punc(self):
        punc_config = self.config.get("punc")
        if not punc_config:
            self.punc = None
            return

        punc_model_path = self.model_paths.get("punc")
        if not punc_model_path:
            self.punc = None
            return

        logger.info("Loading Punc model...")
        from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig

        config = FireRedPuncConfig(**punc_config)
        self.punc = FireRedPunc.from_pretrained(punc_model_path, config)
        logger.info("Punc model loaded!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_resources()
        return False

    def release_resources(self):
        if hasattr(self, "vad"):
            del self.vad
            self.vad = None
        if hasattr(self, "asr"):
            del self.asr
            self.asr = None
        if hasattr(self, "punc"):
            del self.punc
            self.punc = None

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("MixedASRSystem resources released")

    @staticmethod
    def _load_audio(audio_path: str) -> np.ndarray:
        audio_data, _ = sf.read(audio_path, dtype="int16")
        return audio_data.astype(np.float32)

    @staticmethod
    def _get_gpu_memory_info() -> Optional[Dict[str, float]]:
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
                return {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "max_allocated_gb": round(max_allocated, 2),
                }
        except ImportError:
            pass
        return None

    @staticmethod
    def _cleanup_resources(force_cuda_cache: bool = True):
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                if force_cuda_cache:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    logger.debug("CUDA cache cleared")
        except ImportError:
            pass

    def _check_memory_pressure(self) -> bool:
        mem_info = self._get_gpu_memory_info()
        if mem_info is None:
            return False

        import torch

        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            usage_ratio = mem_info["allocated_gb"] / total_memory
            if usage_ratio > 0.90:
                logger.warning(
                    f"High GPU memory usage: {mem_info['allocated_gb']:.2f}GB / "
                    f"{total_memory:.2f}GB ({usage_ratio*100:.1f}%)"
                )
                return True
        return False

    @staticmethod
    def _batch_segments_bfd(
        segments: List[VadSegment],
        max_batch_dur_s=64,
        max_bucket_num=5,
        max_ratio=2,
        short_seg_thresh=1.0,
        heap_switch_thresh=500,
    ):
        """
        BFD (Best-Fit Decreasing) 算法：动态聚类分桶 + 降序排序 + 最佳适应贪心 + 短片段填充
        小N使用线性遍历BFD，大N使用堆调度BFD (O(N log M))
        :param segments: VAD片段列表（单个音频）
        :param max_batch_dur_s: ASR单batch最大总时长
        :param max_bucket_num: 最大桶数（避免过细分桶）
        :param max_ratio: 同桶最长/最短时长比
        :param short_seg_thresh: 短片段阈值（秒），≤该值的片段会做填充优化
        :param heap_switch_thresh: 启用堆优化的片段数阈值，N>该值时使用堆调度
        :return: 分批后的片段索引列表
        """
        if not segments:
            return []

        seg_info = []
        for idx, seg in enumerate(segments):
            dur = seg.end_s - seg.start_s
            seg_info.append((idx, seg.start_s, seg.end_s, dur))

        seg_info_sorted = sorted(seg_info, key=lambda x: x[3], reverse=True)

        buckets = []
        for seg in seg_info_sorted:
            idx, s, e, dur = seg
            if not buckets:
                buckets.append([seg])
                continue
            added = False
            for b in buckets:
                b_min_dur = min(x[3] for x in b)
                if dur / b_min_dur <= max_ratio:
                    b.append(seg)
                    added = True
                    break
            if not added:
                if len(buckets) < max_bucket_num:
                    buckets.append([seg])
                else:
                    buckets[-1].append(seg)

        total_seg_num = len(seg_info)
        use_heap = total_seg_num > heap_switch_thresh

        all_batches = []
        for bucket in buckets:
            if not bucket:
                continue

            if use_heap:
                batch_state: Dict[int, Tuple[float, float, List[int]]] = {}
                min_heap: List[Tuple[float, int]] = []
                batch_counter = 0

                for seg in bucket:
                    idx, s, e, dur = seg
                    target_batch_id = -1
                    valid_entries = []

                    while min_heap:
                        current_R, b_id = heapq.heappop(min_heap)
                        actual_valid_dur, _, _ = batch_state[b_id]
                        actual_R = max_batch_dur_s - actual_valid_dur
                        if abs(current_R - actual_R) > 1e-6:
                            continue
                        valid_entries.append((current_R, b_id))
                        if current_R >= dur:
                            target_batch_id = b_id
                            break

                    for entry in valid_entries:
                        heapq.heappush(min_heap, entry)

                    if target_batch_id != -1:
                        valid_dur, max_len, seg_idxs = batch_state[target_batch_id]
                        new_valid_dur = valid_dur + dur
                        new_max_len = max(max_len, dur)
                        new_seg_idxs = seg_idxs + [idx]
                        batch_state[target_batch_id] = (
                            new_valid_dur,
                            new_max_len,
                            new_seg_idxs,
                        )
                        new_R = max_batch_dur_s - new_valid_dur
                        heapq.heappush(min_heap, (new_R, target_batch_id))
                    else:
                        new_batch_id = batch_counter
                        batch_counter += 1
                        batch_state[new_batch_id] = (dur, dur, [idx])
                        new_R = max_batch_dur_s - dur
                        heapq.heappush(min_heap, (new_R, new_batch_id))

                batches = [(state[0], state[2]) for state in batch_state.values()]
            else:
                batches = []
                for seg in bucket:
                    idx, s, e, dur = seg
                    min_remain = float("inf")
                    target_idx = -1
                    for b_idx, (used_dur, seg_idxs) in enumerate(batches):
                        remain = max_batch_dur_s - used_dur
                        if remain >= dur and remain < min_remain:
                            min_remain = remain
                            target_idx = b_idx
                    if target_idx != -1:
                        batches[target_idx] = (
                            batches[target_idx][0] + dur,
                            batches[target_idx][1] + [idx],
                        )
                    else:
                        batches.append((dur, [idx]))

            if short_seg_thresh > 0:
                short_segs = []
                long_batches = []
                for used_dur, seg_idxs in batches:
                    if len(seg_idxs) == 1:
                        seg = [s for s in bucket if s[0] == seg_idxs[0]][0]
                        if seg[3] <= short_seg_thresh:
                            short_segs.append((seg_idxs[0], seg[3]))
                            continue
                    long_batches.append((used_dur, seg_idxs))
                for seg_idx, dur in short_segs:
                    filled = False
                    for b_idx, (used_dur, seg_idxs) in enumerate(long_batches):
                        remain = max_batch_dur_s - used_dur
                        if remain >= dur:
                            long_batches[b_idx] = (used_dur + dur, seg_idxs + [seg_idx])
                            filled = True
                            break
                    if not filled:
                        long_batches.append((dur, [seg_idx]))
                batches = long_batches

            all_batches.extend([b[1] for b in batches])

        return all_batches

    @staticmethod
    def _batch_segments_wfd(
        segments: List[VadSegment], max_batch_dur_s=64, max_bucket_num=5, max_ratio=2
    ):
        """
        WFD (Worst-Fit Decreasing) 算法：基于最小堆的 O(N log M) 装箱调度
        :param segments: VAD片段列表（单个音频）
        :param max_batch_dur_s: ASR单batch最大总时长限制
        :param max_bucket_num: 最大桶数（避免长短音频悬殊导致过大 Padding）
        :param max_ratio: 同桶最长/最短时长比
        :return: 分批后的片段索引列表，形如 [[idx1, idx2], [idx3], ...]
        """

        if not segments:
            return []

        seg_info = [
            (idx, seg.start_s, seg.end_s, seg.end_s - seg.start_s)
            for idx, seg in enumerate(segments)
        ]
        seg_info_sorted = sorted(seg_info, key=lambda x: x[3], reverse=True)

        buckets = []
        for seg in seg_info_sorted:
            idx, s, e, dur = seg
            added = False
            if buckets:
                for b in buckets:
                    b_min_dur = b[-1][3]
                    if dur / b_min_dur <= max_ratio:
                        b.append(seg)
                        added = True
                        break

            if not added:
                if len(buckets) < max_bucket_num or not buckets:
                    buckets.append([seg])
                else:
                    buckets[-1].append(seg)

        all_batches = []

        for bucket in buckets:
            if not bucket:
                continue

            min_heap = []
            batch_counter = 0

            for seg in bucket:
                idx, _, _, dur = seg

                if min_heap and min_heap[0][0] + dur <= max_batch_dur_s:
                    used_dur, b_id, seg_idxs = heapq.heappop(min_heap)
                    seg_idxs.append(idx)
                    heapq.heappush(min_heap, (used_dur + dur, b_id, seg_idxs))
                else:
                    heapq.heappush(min_heap, (dur, batch_counter, [idx]))
                    batch_counter += 1

            for used_dur, b_id, seg_idxs in min_heap:
                all_batches.append(seg_idxs)

        return all_batches

    @staticmethod
    def batch_vad_segments(
        segments: List[VadSegment],
        max_batch_dur_s=64,
        max_bucket_num=5,
        max_ratio=2,
        use_bfd=False,
    ):
        """
        VAD 片段分批算法入口
        :param segments: VAD片段列表（单个音频）
        :param max_batch_dur_s: ASR单batch最大总时长
        :param max_bucket_num: 最大桶数
        :param max_ratio: 同桶最长/最短时长比
        :param use_bfd: 是否使用 BFD 算法（默认使用 WFD）
        :return: 分批后的片段索引列表
        """
        if use_bfd:
            return MixedASRSystem._batch_segments_bfd(
                segments, max_batch_dur_s, max_bucket_num, max_ratio
            )
        else:
            return MixedASRSystem._batch_segments_wfd(
                segments, max_batch_dur_s, max_bucket_num, max_ratio
            )

    @staticmethod
    def _subdivide_batch_by_duration(
        audio_durs: List[float], target_dur: float
    ) -> List[List[int]]:
        """
        按时长分割 batch，使用贪心算法
        :param audio_durs: 音频时长列表
        :param target_dur: 目标子batch最大时长
        :return: 分割后的索引列表，[[idx1, idx2], [idx3], ...]
        """
        if not audio_durs:
            return []

        indexed_durs = [(i, d) for i, d in enumerate(audio_durs)]
        indexed_durs.sort(key=lambda x: x[1], reverse=True)

        sub_batches = []
        for idx, dur in indexed_durs:
            placed = False
            for sub_batch in sub_batches:
                sub_batch_dur = sum(audio_durs[i] for i in sub_batch)
                if sub_batch_dur + dur <= target_dur:
                    sub_batch.append(idx)
                    placed = True
                    break
            if not placed:
                sub_batches.append([idx])

        return sub_batches

    def _transcribe_batch_with_fallback(
        self,
        uttids: List[str],
        wavs: List[Tuple],
        audio_durs: List[float],
        current_max_dur: Optional[float] = None,
        min_dur_s: float = 8.0,
    ) -> Tuple[Dict, Dict, int]:
        """
        支持自适应降级的 batch 处理函数（按时长降级）

        Args:
            uttids: utterance ID 列表
            wavs: 音频数据列表，格式 [(sample_rate, wav_np), ...]
            audio_durs: 原始音频时长列表
            current_max_dur: 当前子batch最大时长限制（OOM降级时使用）
            min_dur_s: 最小时长阈值，低于此值将逐个处理

        Returns:
            (uttid2result, uttid2audio_dur, failed_count)
        """
        uttid2result = {}
        uttid2audio_dur = {}
        failed_count = 0

        try:
            batch_asr_results = self.asr.transcribe(uttids, wavs)
            batch_asr_results = [
                a for a in batch_asr_results if not _BLANK_SIL_PATTERN.search(a["text"])
            ]

            for res, audio_dur in zip(batch_asr_results, audio_durs):
                uttid = res["uttid"]
                if "timestamp" in res:
                    res["timestamp"] = filter_timestamp_by_duration(
                        res["timestamp"], audio_dur
                    )
                uttid2result[uttid] = res
                uttid2audio_dur[uttid] = audio_dur

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                total_dur = sum(audio_durs)
                logger.warning(
                    f"OOM with batch_size={len(uttids)}, total_dur={total_dur:.1f}s, falling back..."
                )

                import gc

                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                if total_dur <= min_dur_s or len(uttids) == 1:
                    logger.error(
                        f"Cannot reduce further (total_dur={total_dur:.1f}s <= "
                        f"min_dur={min_dur_s}s), processing one by one..."
                    )
                    for i in range(len(uttids)):
                        try:
                            single_result = self.asr.transcribe([uttids[i]], [wavs[i]])
                            single_result = [
                                a
                                for a in single_result
                                if not _BLANK_SIL_PATTERN.search(a["text"])
                            ]
                            if single_result:
                                res = single_result[0]
                                if "timestamp" in res:
                                    res["timestamp"] = filter_timestamp_by_duration(
                                        res["timestamp"], audio_durs[i]
                                    )
                                uttid2result[uttids[i]] = res
                                uttid2audio_dur[uttids[i]] = audio_durs[i]
                        except Exception as inner_e:
                            logger.error(
                                f"Failed to process single segment {uttids[i]}: {inner_e}"
                            )
                            failed_count += 1
                    return uttid2result, uttid2audio_dur, failed_count

                target_dur = current_max_dur or (total_dur // 2)
                target_dur = max(target_dur, min_dur_s)
                logger.info(f"Retrying with target_dur={target_dur:.1f}s")

                sub_batch_indices = self._subdivide_batch_by_duration(
                    audio_durs, target_dur
                )

                for sub_indices in sub_batch_indices:
                    sub_uttids = [uttids[i] for i in sub_indices]
                    sub_wavs = [wavs[i] for i in sub_indices]
                    sub_audio_durs = [audio_durs[i] for i in sub_indices]

                    sub_result, sub_dur, sub_failed = (
                        self._transcribe_batch_with_fallback(
                            sub_uttids,
                            sub_wavs,
                            sub_audio_durs,
                            current_max_dur=target_dur,
                            min_dur_s=min_dur_s,
                        )
                    )

                    uttid2result.update(sub_result)
                    uttid2audio_dur.update(sub_dur)
                    failed_count += sub_failed
            else:
                logger.error(f"Non-OOM RuntimeError: {str(e)}", exc_info=True)
                failed_count = len(uttids)
        except Exception as e:
            logger.error(f"Unexpected error during ASR: {str(e)}", exc_info=True)
            failed_count = len(uttids)

        return uttid2result, uttid2audio_dur, failed_count

    def _transcribe_batch_from_cache_with_fallback(
        self,
        cached_fbank,
        segment_infos: List[Tuple[int, int, str, float]],
        audio_durs: List[float],
        current_max_dur: Optional[float] = None,
        min_dur_s: float = 8.0,
    ) -> Tuple[Dict, Dict, int]:
        """
        从缓存的fbank特征进行ASR推理，支持自适应降级

        Args:
            cached_fbank: VAD阶段缓存的fbank特征 tensor [T, 80]
            segment_infos: 片段信息列表 [(start_frame, end_frame, uttid, dur), ...]
            audio_durs: 原始音频时长列表
            current_max_dur: 当前子batch最大时长限制（OOM降级时使用）
            min_dur_s: 最小时长阈值，低于此值将逐个处理

        Returns:
            (uttid2result, uttid2audio_dur, failed_count)
        """
        uttid2result = {}
        uttid2audio_dur = {}
        failed_count = 0

        try:
            batch_asr_results = self.asr.transcribe_from_cached_fbank(
                cached_fbank, segment_infos
            )
            batch_asr_results = [
                a for a in batch_asr_results if not _BLANK_SIL_PATTERN.search(a["text"])
            ]

            for res, audio_dur in zip(batch_asr_results, audio_durs):
                uttid = res["uttid"]
                if "timestamp" in res:
                    res["timestamp"] = filter_timestamp_by_duration(
                        res["timestamp"], audio_dur
                    )
                uttid2result[uttid] = res
                uttid2audio_dur[uttid] = audio_dur

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                total_dur = sum(audio_durs)
                logger.warning(
                    f"OOM with batch_size={len(segment_infos)}, total_dur={total_dur:.1f}s, falling back..."
                )

                import gc

                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                if total_dur <= min_dur_s or len(segment_infos) == 1:
                    logger.error(
                        f"Cannot reduce further (total_dur={total_dur:.1f}s <= "
                        f"min_dur={min_dur_s}s), processing one by one..."
                    )
                    for i in range(len(segment_infos)):
                        try:
                            single_result = self.asr.transcribe_from_cached_fbank(
                                cached_fbank, [segment_infos[i]]
                            )
                            single_result = [
                                a
                                for a in single_result
                                if not _BLANK_SIL_PATTERN.search(a["text"])
                            ]
                            if single_result:
                                res = single_result[0]
                                if "timestamp" in res:
                                    res["timestamp"] = filter_timestamp_by_duration(
                                        res["timestamp"], audio_durs[i]
                                    )
                                uttid2result[segment_infos[i][2]] = res
                                uttid2audio_dur[segment_infos[i][2]] = audio_durs[i]
                        except Exception as inner_e:
                            logger.error(
                                f"Failed to process single segment {segment_infos[i][2]}: {inner_e}"
                            )
                            failed_count += 1
                    return uttid2result, uttid2audio_dur, failed_count

                target_dur = current_max_dur or (total_dur // 2)
                target_dur = max(target_dur, min_dur_s)
                logger.info(f"Retrying with target_dur={target_dur:.1f}s")

                sub_batch_indices = self._subdivide_batch_by_duration(
                    audio_durs, target_dur
                )

                for sub_indices in sub_batch_indices:
                    sub_segment_infos = [segment_infos[i] for i in sub_indices]
                    sub_audio_durs = [audio_durs[i] for i in sub_indices]

                    sub_result, sub_dur, sub_failed = (
                        self._transcribe_batch_from_cache_with_fallback(
                            cached_fbank,
                            sub_segment_infos,
                            sub_audio_durs,
                            current_max_dur=target_dur,
                            min_dur_s=min_dur_s,
                        )
                    )

                    uttid2result.update(sub_result)
                    uttid2audio_dur.update(sub_dur)
                    failed_count += sub_failed
            else:
                logger.error(f"Non-OOM RuntimeError: {str(e)}", exc_info=True)
                failed_count = len(segment_infos)
        except Exception as e:
            logger.error(f"Unexpected error during ASR: {str(e)}", exc_info=True)
            failed_count = len(segment_infos)

        return uttid2result, uttid2audio_dur, failed_count

    def _process_single_audio(
        self, audio_idx: int, audio_info: Dict, max_batch_dur_s: float
    ) -> AudioProcessResult:
        """
        处理单个音频：加载 -> VAD -> ASR -> 合并结果

        Args:
            audio_idx: 音频索引
            audio_info: 音频信息字典
            max_batch_dur_s: ASR batch 最大时长

        Returns:
            AudioProcessResult: 处理结果
        """
        audio_path = audio_info["audio_path"]
        original_file = audio_info.get("original_file")
        start_time = time.perf_counter()
        audio_data = None

        try:
            logger.info(f"[{audio_idx}] 开始处理音频: {audio_path}")

            audio_data = self._load_audio(audio_path)
            dur_s = audio_data.shape[0] / SAMPLE_RATE
            logger.debug(f"[{audio_idx}] 音频加载完成，时长: {dur_s:.2f}s")

            vad_result, probs, cached_fbank = self.vad.detect(
                (SAMPLE_RATE, audio_data), return_fbank_cache=True
            )
            del probs
            logger.debug(
                f"[{audio_idx}] VAD fbank缓存已启用，shape: {cached_fbank.shape if cached_fbank is not None else 'None'}"
            )

            del audio_data
            audio_data = None
            gc.collect()

            if vad_result is None:
                logger.warning(f"[{audio_idx}] VAD 结果为空，跳过")
                return AudioProcessResult(
                    audio_idx=audio_idx,
                    audio_path=audio_path,
                    success=False,
                    error_msg="VAD result is empty",
                )

            vad_segments = vad_result["timestamps"]
            logger.info(
                f"[{audio_idx}] VAD 完成，检测到 {len(vad_segments)} 个有效片段"
            )

            audio_segments = []
            for seg_idx, (start_s, end_s) in enumerate(vad_segments):
                uttid = f"audio{audio_idx}_seg{seg_idx}_s{round(start_s*1000)}_e{round(end_s*1000)}"
                audio_segments.append(
                    VadSegment(
                        seg_idx=seg_idx, start_s=start_s, end_s=end_s, uttid=uttid
                    )
                )

            if not audio_segments:
                logger.warning(f"[{audio_idx}] 没有有效VAD片段，跳过")
                return AudioProcessResult(
                    audio_idx=audio_idx,
                    audio_path=audio_path,
                    success=False,
                    error_msg="No valid VAD segments",
                )

            batches = self.batch_vad_segments(
                audio_segments, max_batch_dur_s=max_batch_dur_s, use_bfd=self.use_bfd
            )
            logger.info(f"[{audio_idx}] 分批完成，共 {len(batches)} 个 batch")

            uttid2asr_result = {}
            failed_batches = 0

            logger.info(f"[{audio_idx}] 开始 ASR 推理，共 {len(batches)} 个 batch")

            FRAME_SHIFT_S = 0.010

            for batch_idx, seg_indices in enumerate(batches):
                try:
                    batch_segs = [audio_segments[i] for i in seg_indices]

                    segment_infos = []
                    batch_audio_durs = []

                    for seg in batch_segs:
                        start_frame = int(seg.start_s / FRAME_SHIFT_S)
                        end_frame = int(seg.end_s / FRAME_SHIFT_S)
                        dur = seg.end_s - seg.start_s

                        segment_infos.append((start_frame, end_frame, seg.uttid, dur))
                        batch_audio_durs.append(dur)

                    logger.debug(
                        f"[{audio_idx}] ASR batch {batch_idx+1}/{len(batches)}: {len(segment_infos)} segments"
                    )

                    result, _, failed = self._transcribe_batch_from_cache_with_fallback(
                        cached_fbank, segment_infos, batch_audio_durs
                    )

                    uttid2asr_result.update(result)
                    failed_batches += failed

                    logger.info(
                        f"[{audio_idx}] ASR batch {batch_idx+1}/{len(batches)} 完成，{len(segment_infos)} 个片段"
                    )

                    del segment_infos
                    del batch_audio_durs
                    del batch_segs

                    gc.collect()

                except Exception as e:
                    logger.error(
                        f"[{audio_idx}] Batch {batch_idx+1} 处理异常: {str(e)}",
                        exc_info=True,
                    )
                    failed_batches += 1

            del cached_fbank
            cached_fbank = None
            self._cleanup_resources()

            logger.info(
                f"[{audio_idx}] ASR 完成，成功 {len(uttid2asr_result)} 个片段，失败 {failed_batches} 个片段"
            )

            sentences = []
            text_parts = []
            words = []

            for seg in audio_segments:
                if seg.uttid not in uttid2asr_result:
                    continue

                asr_res = uttid2asr_result[seg.uttid]
                start_ms = round(seg.start_s * 1000)
                end_ms = round(seg.end_s * 1000)
                text = asr_res.get("text", "")
                text_parts.append(text)

                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_res.get("confidence", 0.0),
                }
                sentences.append(sentence)

                if "timestamp" in asr_res:
                    for w, s, e in asr_res["timestamp"]:
                        start_ms_word = max(0, round(s * 1000 + start_ms))
                        end_ms_word = max(start_ms_word, round(e * 1000 + start_ms))
                        words.append(
                            {
                                "start_ms": start_ms_word,
                                "end_ms": end_ms_word,
                                "text": w,
                            }
                        )

            full_text = "".join(text_parts)

            if self.punc and sentences:
                logger.info(f"[{audio_idx}] 开始添加标点...")
                punc_error = None

                try:
                    if words:
                        batch_timestamp = []
                        expected_punc_count = 0
                        for seg in audio_segments:
                            if seg.uttid not in uttid2asr_result:
                                continue
                            asr_res = uttid2asr_result[seg.uttid]
                            if "timestamp" in asr_res and asr_res["timestamp"]:
                                seg_start_ms = round(seg.start_s * 1000)
                                timestamp_list = []
                                for w, s, e in asr_res["timestamp"]:
                                    timestamp_list.append((w, s, e))
                                batch_timestamp.append(timestamp_list)
                                expected_punc_count += 1

                        if batch_timestamp:
                            PUNC_BATCH_SIZE = 10
                            all_punc_results = []
                            for i in range(0, len(batch_timestamp), PUNC_BATCH_SIZE):
                                batch_chunk = batch_timestamp[i : i + PUNC_BATCH_SIZE]
                                chunk_results = self.punc.process_with_timestamp(
                                    batch_chunk
                                )
                                all_punc_results.extend(chunk_results)
                                del chunk_results
                            punc_results = all_punc_results

                            if len(punc_results) != expected_punc_count:
                                punc_error = f"Punc结果数量({len(punc_results)})与预期({expected_punc_count})不匹配"
                                logger.warning(
                                    f"[{audio_idx}] {punc_error}，保留原始文本"
                                )
                            else:
                                new_sentences = []
                                punc_idx = 0
                                for seg in audio_segments:
                                    if seg.uttid not in uttid2asr_result:
                                        continue
                                    asr_res = uttid2asr_result[seg.uttid]
                                    if (
                                        "timestamp" not in asr_res
                                        or not asr_res["timestamp"]
                                    ):
                                        continue

                                    seg_start_ms = round(seg.start_s * 1000)
                                    seg_end_ms = round(seg.end_s * 1000)

                                    if punc_idx < len(punc_results):
                                        punc_result = punc_results[punc_idx]
                                        punc_sentences = punc_result.get(
                                            "punc_sentences", []
                                        )

                                        for i, punc_sent in enumerate(punc_sentences):
                                            punc_text = punc_sent["punc_text"]
                                            punc_start_s = punc_sent["start_s"]
                                            punc_end_s = punc_sent["end_s"]

                                            new_start_ms = round(
                                                punc_start_s * 1000 + seg_start_ms
                                            )
                                            new_end_ms = round(
                                                punc_end_s * 1000 + seg_start_ms
                                            )

                                            if i == 0:
                                                new_start_ms = seg_start_ms
                                            if i == len(punc_sentences) - 1:
                                                new_end_ms = seg_end_ms

                                            new_sentences.append(
                                                {
                                                    "start_ms": new_start_ms,
                                                    "end_ms": new_end_ms,
                                                    "text": punc_text,
                                                    "asr_confidence": asr_res.get(
                                                        "confidence", 0.0
                                                    ),
                                                }
                                            )
                                        punc_idx += 1

                                if new_sentences:
                                    sentences = new_sentences
                                    full_text = "".join(s["text"] for s in sentences)
                    else:
                        batch_text = [s["text"] for s in sentences]
                        PUNC_BATCH_SIZE = 50
                        all_punc_results = []
                        for i in range(0, len(batch_text), PUNC_BATCH_SIZE):
                            batch_chunk = batch_text[i : i + PUNC_BATCH_SIZE]
                            chunk_results = self.punc.process(batch_chunk)
                            all_punc_results.extend(chunk_results)
                            del chunk_results
                        punc_results = all_punc_results

                        if len(punc_results) != len(sentences):
                            punc_error = f"Punc结果数量({len(punc_results)})与句子数量({len(sentences)})不匹配"
                            logger.warning(f"[{audio_idx}] {punc_error}，保留原始文本")
                        else:
                            for i, punc_res in enumerate(punc_results):
                                sentences[i]["text"] = punc_res["punc_text"]
                            full_text = "".join(s["text"] for s in sentences)

                except Exception as e:
                    punc_error = str(e)
                    logger.warning(
                        f"[{audio_idx}] 标点添加失败: {punc_error}，使用原始文本",
                        exc_info=True,
                    )

                if punc_error is None:
                    logger.info(f"[{audio_idx}] 标点添加完成")

            full_text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", full_text)

            result = {
                "uttid": os.path.basename(audio_path).split(".")[0],
                "text": full_text,
                "sentences": sentences,
                "vad_segments_ms": [
                    (round(seg.start_s * 1000), round(seg.end_s * 1000))
                    for seg in audio_segments
                ],
                "dur_s": dur_s,
                "words": words,
                "wav_path": audio_path,
                "original_file": original_file,
            }

            processing_time = time.perf_counter() - start_time
            rtf = processing_time / dur_s if dur_s > 0 else 0

            logger.info(
                f"[{audio_idx}] 处理完成，耗时: {processing_time:.2f}s, RTF: {rtf:.3f}x"
            )

            return AudioProcessResult(
                audio_idx=audio_idx,
                audio_path=audio_path,
                success=True,
                result=result,
                processing_time_s=processing_time,
                rtf=rtf,
            )

        except Exception as e:
            error_msg = f"处理音频失败: {str(e)}"
            logger.error(f"[{audio_idx}] {error_msg}", exc_info=True)
            return AudioProcessResult(
                audio_idx=audio_idx,
                audio_path=audio_path,
                success=False,
                error_msg=error_msg,
            )
        finally:
            if audio_data is not None:
                del audio_data
            if "cached_fbank" in locals() and cached_fbank is not None:
                del cached_fbank
            force_cache = self._check_memory_pressure()
            self._cleanup_resources(force_cuda_cache=force_cache)

    def batch_process(
        self,
        audio_files: List[Dict],
        max_batch_dur_s: int = 64,
        on_result_callback=None,
    ):
        """
        逐个处理音频文件，支持流式回调

        Args:
            audio_files: 音频文件信息列表
            max_batch_dur_s: ASR batch 最大时长
            on_result_callback: 处理完成回调函数，签名: (audio_idx, result, error_msg) -> None
                                result 为 AudioProcessResult 对象

        Returns:
            stats: 统计信息（流式累加，不存储中间结果）
        """
        if not audio_files:
            return {}

        total_audio_dur = 0.0
        total_processing_time = 0.0
        rtfs = []
        success_count = 0
        error_count = 0

        for audio_idx, audio_info in enumerate(audio_files):
            result = self._process_single_audio(
                audio_idx=audio_idx,
                audio_info=audio_info,
                max_batch_dur_s=max_batch_dur_s,
            )

            if result.success and result.result:
                success_count += 1
                rtfs.append(result.rtf)
                total_audio_dur += result.result["dur_s"]
                total_processing_time += result.processing_time_s
            else:
                error_count += 1
                logger.warning(f"音频 {result.audio_path} 处理失败: {result.error_msg}")

            if on_result_callback:
                on_result_callback(audio_idx, result, len(audio_files))

            if result.result:
                result.result.clear()
            del result

        avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0
        overall_rtf = (
            total_processing_time / total_audio_dur if total_audio_dur > 0 else 0.0
        )

        stats = {
            "total_audio_dur_s": total_audio_dur,
            "total_processing_s": total_processing_time,
            "avg_rtf": avg_rtf,
            "overall_rtf": overall_rtf,
            "success_count": success_count,
            "error_count": error_count,
            "per_audio_rtfs": rtfs,
        }

        return stats
