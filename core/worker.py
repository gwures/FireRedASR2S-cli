import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_LENGTH_SAMPLE = 400
FRAME_SHIFT_SAMPLE = 160

_BLANK_SIL_PATTERN = re.compile(r"(<blank>)|(<sil>)")


def calc_original_seq_len(original_audio_samples: int) -> int:
    if original_audio_samples < FRAME_LENGTH_SAMPLE:
        return 1
    return (original_audio_samples - FRAME_LENGTH_SAMPLE) // FRAME_SHIFT_SAMPLE + 1


def filter_timestamp_by_original_len(timestamp: List, original_seq_len: int, original_audio_dur: float) -> List:
    if not timestamp:
        return timestamp
    
    def frame2time(frame_idx: int) -> float:
        return (frame_idx * FRAME_SHIFT_SAMPLE) / SAMPLE_RATE
    
    filtered_timestamp = []
    for item in timestamp:
        if len(item) != 3:
            filtered_timestamp.append(item)
            continue
        w, s, e = item
        
        s_time = frame2time(s) if isinstance(s, int) else float(s)
        e_time = frame2time(e) if isinstance(e, int) else float(e)
        
        if e_time <= original_audio_dur:
            filtered_timestamp.append((w, s_time, e_time))
        elif s_time < original_audio_dur:
            filtered_timestamp.append((w, s_time, original_audio_dur))
    
    return filtered_timestamp


@dataclass
class VadSegment:
    audio_path: str
    audio_idx: int
    seg_idx: int
    start_s: float
    end_s: float
    uttid: str


@dataclass
class AudioInfo:
    audio_path: str
    audio_idx: int
    original_file: Optional[Path]
    dur_s: float
    segments: List[VadSegment]
    asr_results: List[Dict]
    audio_data: Optional[np.ndarray] = None


def _fix_pkg_resources():
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            import pkg_resources
        except ImportError:
            try:
                from setuptools.extern import pkg_resources
            except ImportError:
                import types
                from importlib.metadata import version as get_version
                pkg_resources = types.ModuleType('pkg_resources')
                pkg_resources.get_distribution = lambda name: types.SimpleNamespace(project_name=name, version=get_version(name))
                pkg_resources.Distribution = type('Distribution', (), {'__init__': lambda self, name, version: setattr(self, 'project_name', name) or setattr(self, 'version', version)})
                sys.modules['pkg_resources'] = pkg_resources

_fix_pkg_resources()

logger = logging.getLogger("fireredasr2s.worker")


class MixedASRSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_paths = config["model_paths"]
        
        self.vad = None
        self.asr = None
        
        self.max_batch_dur_s = config.get("max_batch_dur_s", 64)
        self.max_vad_batch_size = config.get("max_vad_batch_size", 1)
        
        self._audio_cache = {}
        self._audio_cache_max_mb = config.get("audio_cache_max_mb", 4096)
        self._audio_cache_size_mb = 0.0
        
        import threading
        self._cache_lock = threading.Lock()
        
        logger.info(f"MixedASRSystem created (max_batch_dur_s={self.max_batch_dur_s}, max_vad_batch_size={self.max_vad_batch_size}, audio_cache_max_mb={self._audio_cache_max_mb}MB)")
        
        self._load_vad()
        self._load_asr()
        
        logger.info("VAD and ASR preloaded!")
    
    def _load_vad(self):
        if self.vad is not None:
            return
        logger.info("Loading VAD model...")
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig
        vad_config = FireRedVadConfig(**self.config["vad"])
        self.vad = FireRedVad.from_pretrained(self.model_paths["vad"], vad_config)
        logger.info("VAD model loaded!")
    
    def _load_asr(self):
        if self.asr is not None:
            return
        logger.info("Loading ASR model...")
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        asr_config = FireRedAsr2Config(**self.config["asr"])
        self.asr = FireRedAsr2.from_pretrained("aed", self.model_paths["asr"], asr_config)
        logger.info("ASR model loaded!")
    
    def _get_audio_from_cache(self, audio_path: str) -> Optional[np.ndarray]:
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        return None
    
    def _load_audio_to_cache(self, audio_path: str) -> np.ndarray:
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        
        audio_data, sample_rate = sf.read(audio_path, dtype="int16")
        assert sample_rate == 16000, f"{audio_path} 采样率不是16000Hz"
        
        mem_mb = audio_data.nbytes / (1024 * 1024)
        
        with self._cache_lock:
            if audio_path in self._audio_cache:
                return self._audio_cache[audio_path]
            
            while self._audio_cache_size_mb + mem_mb > self._audio_cache_max_mb and self._audio_cache:
                oldest_key = next(iter(self._audio_cache))
                oldest_data = self._audio_cache.pop(oldest_key)
                self._audio_cache_size_mb -= oldest_data.nbytes / (1024 * 1024)
            
            self._audio_cache[audio_path] = audio_data
            self._audio_cache_size_mb += mem_mb
        
        logger.debug(f"加载音频到缓存: {audio_path} ({mem_mb:.1f}MB), 缓存大小: {self._audio_cache_size_mb:.1f}MB")
        
        return audio_data
    
    def _get_audio_slice(self, audio_path: str, start_sample: int, end_sample: int) -> np.ndarray:
        cached_audio = self._get_audio_from_cache(audio_path)
        if cached_audio is not None:
            return cached_audio[start_sample:end_sample]
        return None
    
    def _preload_audios(self, audio_paths: List[str]):
        for audio_path in audio_paths:
            if audio_path not in self._audio_cache:
                self._load_audio_to_cache(audio_path)
    
    def _clear_audio_cache(self):
        self._audio_cache.clear()
        self._audio_cache_size_mb = 0.0
        logger.info("音频缓存已释放")
    
    @staticmethod
    def split_vad_to_batches(all_segments: List[VadSegment], max_batch_dur_s=64, max_bucket_num=4, max_ratio=3, short_seg_thresh=1.0):
        """
        动态聚类分桶 + 降序排序+最佳适应贪心 + 短片段填充
        按音频分组后，对每个音频单独分批，保留数据局部性，提升读取效率
        :param all_segments: 全局VAD片段列表，List[VadSegment]
        :param max_batch_dur_s: ASR单batch最大总时长
        :param max_bucket_num: 最大桶数（避免过细分桶）
        :param max_ratio: 同桶最长/最短时长比
        :param short_seg_thresh: 短片段阈值（秒），≤该值的片段会做填充优化
        :return: 分批后的片段索引列表
        """
        if not all_segments:
            return []
        
        audio_to_segs = {}
        for idx, seg in enumerate(all_segments):
            audio_idx = seg.audio_idx
            if audio_idx not in audio_to_segs:
                audio_to_segs[audio_idx] = []
            audio_to_segs[audio_idx].append((idx, seg.start_s, seg.end_s))
        
        def _split_single_audio(vad_segments, max_batch_dur_s, max_bucket_num, max_ratio, short_seg_thresh):
            if not vad_segments:
                return []
            
            seg_info = []
            for idx, s, e in vad_segments:
                dur = e - s
                seg_info.append((idx, s, e, dur))
            
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
            
            all_batches = []
            for bucket in buckets:
                if not bucket:
                    continue
                batches = []
                for seg in bucket:
                    idx, s, e, dur = seg
                    min_remain = float('inf')
                    target_idx = -1
                    for b_idx, (used_dur, seg_idxs) in enumerate(batches):
                        remain = max_batch_dur_s - used_dur
                        if remain >= dur and remain < min_remain:
                            min_remain = remain
                            target_idx = b_idx
                    if target_idx != -1:
                        batches[target_idx] = (batches[target_idx][0] + dur, batches[target_idx][1] + [idx])
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
        
        global_batches = []
        for audio_idx in sorted(audio_to_segs.keys()):
            segs = audio_to_segs[audio_idx]
            audio_batches = _split_single_audio(segs, max_batch_dur_s, max_bucket_num, max_ratio, short_seg_thresh)
            global_batches.extend(audio_batches)
        
        return global_batches

    @staticmethod
    def _split_batch_by_duration(audio_durs: List[float], target_dur: float) -> List[List[int]]:
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

    def _process_batch_with_adaptive_fallback(
        self,
        uttids: List[str],
        wavs: List[Tuple],
        seq_lens: List[int],
        audio_durs: List[float],
        current_max_dur: float = None,
        min_dur_s: float = 8.0
    ) -> Tuple[Dict, Dict, Dict, int]:
        """
        支持自适应降级的 batch 处理函数（按时长降级）
        
        Args:
            uttids: utterance ID 列表
            wavs: 音频数据列表，格式 [(sample_rate, wav_np), ...]
            seq_lens: 原始序列长度列表
            audio_durs: 原始音频时长列表
            current_max_dur: 当前子batch最大时长限制（OOM降级时使用）
            min_dur_s: 最小时长阈值，低于此值将逐个处理
            
        Returns:
            (uttid2result, uttid2seq_len, uttid2audio_dur, failed_count)
        """
        uttid2result = {}
        uttid2seq_len = {}
        uttid2audio_dur = {}
        failed_count = 0
        
        try:
            batch_asr_results = self.asr.transcribe(uttids, wavs)
            batch_asr_results = [a for a in batch_asr_results if not _BLANK_SIL_PATTERN.search(a["text"])]
            
            for res, seq_len, audio_dur in zip(batch_asr_results, seq_lens, audio_durs):
                uttid = res["uttid"]
                if "timestamp" in res:
                    res["timestamp"] = filter_timestamp_by_original_len(
                        res["timestamp"], seq_len, audio_dur
                    )
                uttid2result[uttid] = res
                uttid2seq_len[uttid] = seq_len
                uttid2audio_dur[uttid] = audio_dur
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                total_dur = sum(audio_durs)
                logger.warning(f"OOM with batch_size={len(uttids)}, total_dur={total_dur:.1f}s, falling back...")
                
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                if total_dur <= min_dur_s or len(uttids) == 1:
                    logger.error(f"Cannot reduce further (total_dur={total_dur:.1f}s <= min_dur={min_dur_s}s), processing one by one...")
                    for i in range(len(uttids)):
                        try:
                            single_result = self.asr.transcribe([uttids[i]], [wavs[i]])
                            single_result = [a for a in single_result if not _BLANK_SIL_PATTERN.search(a["text"])]
                            if single_result:
                                res = single_result[0]
                                if "timestamp" in res:
                                    res["timestamp"] = filter_timestamp_by_original_len(
                                        res["timestamp"], seq_lens[i], audio_durs[i]
                                    )
                                uttid2result[uttids[i]] = res
                                uttid2seq_len[uttids[i]] = seq_lens[i]
                                uttid2audio_dur[uttids[i]] = audio_durs[i]
                        except Exception as inner_e:
                            logger.error(f"Failed to process single segment {uttids[i]}: {inner_e}")
                            failed_count += 1
                    return uttid2result, uttid2seq_len, uttid2audio_dur, failed_count
                
                target_dur = current_max_dur or (total_dur // 2)
                target_dur = max(target_dur, min_dur_s)
                logger.info(f"Retrying with target_dur={target_dur:.1f}s")
                
                sub_batch_indices = self._split_batch_by_duration(audio_durs, target_dur)
                
                for sub_indices in sub_batch_indices:
                    sub_uttids = [uttids[i] for i in sub_indices]
                    sub_wavs = [wavs[i] for i in sub_indices]
                    sub_seq_lens = [seq_lens[i] for i in sub_indices]
                    sub_audio_durs = [audio_durs[i] for i in sub_indices]
                    
                    sub_result, sub_seq, sub_dur, sub_failed = \
                        self._process_batch_with_adaptive_fallback(
                            sub_uttids, sub_wavs, sub_seq_lens, sub_audio_durs,
                            current_max_dur=target_dur,
                            min_dur_s=min_dur_s
                        )
                    
                    uttid2result.update(sub_result)
                    uttid2seq_len.update(sub_seq)
                    uttid2audio_dur.update(sub_dur)
                    failed_count += sub_failed
            else:
                logger.error(f"Non-OOM RuntimeError: {str(e)}", exc_info=True)
                failed_count = len(uttids)
        except Exception as e:
            logger.error(f"Unexpected error during ASR: {str(e)}", exc_info=True)
            failed_count = len(uttids)
        
        return uttid2result, uttid2seq_len, uttid2audio_dur, failed_count

    def batch_vad_process(self, audio_files: List[Dict], max_workers=8, max_preload_workers=4):
        audio_paths = [info["audio_path"] for info in audio_files]
        
        self._clear_audio_cache()
        
        preload_start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=max_preload_workers) as executor:
            future_to_path = {
                executor.submit(self._load_audio_to_cache, path): path
                for path in audio_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"预加载音频失败: {path}, error: {str(e)}")
        
        preload_time = time.perf_counter() - preload_start
        logger.info(f"音频预加载完成，耗时 {preload_time:.2f}s")
        
        all_segments = []
        audio_info_map = {}
        
        def create_audio_segments(audio_idx: int, audio_info: Dict, vad_result: Dict) -> Tuple[int, List[VadSegment], AudioInfo]:
            audio_path = audio_info["audio_path"]
            original_file = audio_info.get("original_file")
            
            audio_data = self._get_audio_from_cache(audio_path)
            dur = audio_data.shape[0] / SAMPLE_RATE
            
            vad_segments = vad_result["timestamps"]
            
            audio_segments = []
            for seg_idx, (start_s, end_s) in enumerate(vad_segments):
                uttid = f"audio{audio_idx}_seg{seg_idx}_s{round(start_s*1000)}_e{round(end_s*1000)}"
                audio_segments.append(VadSegment(
                    audio_path=audio_path,
                    audio_idx=audio_idx,
                    seg_idx=seg_idx,
                    start_s=start_s,
                    end_s=end_s,
                    uttid=uttid
                ))
            
            audio_info_obj = AudioInfo(
                audio_path=audio_path,
                audio_idx=audio_idx,
                original_file=original_file,
                dur_s=dur,
                segments=audio_segments,
                asr_results=[],
                audio_data=None
            )
            
            return audio_idx, audio_segments, audio_info_obj

        logger.info(f"批量VAD开始，共{len(audio_files)}个音频，batch_size={self.max_vad_batch_size}")
        
        for batch_start in range(0, len(audio_files), self.max_vad_batch_size):
            batch_end = min(batch_start + self.max_vad_batch_size, len(audio_files))
            batch_items = audio_files[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            try:
                batch_audios = []
                for info in batch_items:
                    audio_path = info["audio_path"]
                    audio_data = self._get_audio_from_cache(audio_path)
                    if audio_data is None:
                        audio_data = self._load_audio_to_cache(audio_path)
                    batch_audios.append((SAMPLE_RATE, audio_data))
                
                vad_results, _ = self.vad.detect_batch(batch_audios)
                
                for local_idx, (audio_info, vad_result) in enumerate(zip(batch_items, vad_results)):
                    global_idx = batch_indices[local_idx]
                    if vad_result is None:
                        logger.warning(f"音频 {audio_info['audio_path']} VAD 结果为空，跳过")
                        continue
                    
                    try:
                        audio_idx, segments, info_obj = create_audio_segments(global_idx, audio_info, vad_result)
                        audio_info_map[audio_idx] = info_obj
                        all_segments.extend(segments)
                    except Exception as e:
                        logger.error(f"处理音频 {audio_info['audio_path']} 片段失败: {str(e)}", exc_info=True)
                
                logger.info(f"VAD batch [{batch_start+1}-{batch_end}]/{len(audio_files)} 完成")
                
            except Exception as e:
                logger.error(f"VAD batch [{batch_start+1}-{batch_end}] 失败: {str(e)}，降级为逐个处理", exc_info=True)
                
                for local_idx, audio_info in enumerate(batch_items):
                    global_idx = batch_indices[local_idx]
                    try:
                        audio_path = audio_info["audio_path"]
                        audio_data = self._get_audio_from_cache(audio_path)
                        if audio_data is None:
                            audio_data = self._load_audio_to_cache(audio_path)
                        
                        vad_result, _ = self.vad.detect((SAMPLE_RATE, audio_data))
                        
                        audio_idx, segments, info_obj = create_audio_segments(global_idx, audio_info, vad_result)
                        audio_info_map[audio_idx] = info_obj
                        all_segments.extend(segments)
                        
                    except Exception as inner_e:
                        logger.error(f"处理音频 {audio_info['audio_path']} 失败: {str(inner_e)}", exc_info=True)

        logger.info(f"批量VAD完成，共{len(audio_info_map)}个音频，切出{len(all_segments)}个有效片段")
        return all_segments, audio_info_map

    def batch_asr_process(self, all_segments: List[VadSegment], audio_info_map: Dict[int, AudioInfo], max_batch_dur_s=None):
        if max_batch_dur_s is None:
            max_batch_dur_s = self.max_batch_dur_s

        batches = self.split_vad_to_batches(
            all_segments,
            max_batch_dur_s=max_batch_dur_s
        )
        logger.info(f"全局分批完成，共{len(batches)}个batch")

        uttid2asr_result = {}
        uttid2original_seq_len = {}
        uttid2original_audio_dur = {}
        failed_batches = 0
        
        for batch_idx, seg_indices in enumerate(batches):
            try:
                batch_uttids = []
                batch_wavs = []
                batch_segs = [all_segments[i] for i in seg_indices]
                
                valid_batch_data = []
                for seg in batch_segs:
                    try:
                        start_sample = round(seg.start_s * SAMPLE_RATE)
                        end_sample = round(seg.end_s * SAMPLE_RATE)
                        
                        wav_segment = self._get_audio_slice(seg.audio_path, start_sample, end_sample)
                        if wav_segment is None:
                            wav_segment = self._load_audio_to_cache(seg.audio_path)
                            wav_segment = wav_segment[start_sample:end_sample]
                        
                        original_audio_dur = seg.end_s - seg.start_s
                        original_audio_samples = len(wav_segment)
                        original_seq_len = calc_original_seq_len(original_audio_samples)
                        
                        valid_batch_data.append({
                            'uttid': seg.uttid,
                            'wav': (SAMPLE_RATE, wav_segment),
                            'seq_len': original_seq_len,
                            'audio_dur': original_audio_dur,
                            'seg': seg
                        })
                    except Exception as e:
                        logger.warning(f"读取片段失败，跳过: uttid={seg.uttid}, error={str(e)}")
                        continue
                
                if not valid_batch_data:
                    logger.warning(f"Batch {batch_idx+1}/{len(batches)} 没有有效片段，跳过")
                    continue
                
                batch_uttids = [d['uttid'] for d in valid_batch_data]
                batch_wavs = [d['wav'] for d in valid_batch_data]
                batch_original_seq_lens = [d['seq_len'] for d in valid_batch_data]
                batch_original_audio_durs = [d['audio_dur'] for d in valid_batch_data]

                logger.info(f"ASR batch {batch_idx+1}/{len(batches)}: processing {len(batch_wavs)} segments")
                
                result, seq_len, audio_dur, failed = self._process_batch_with_adaptive_fallback(
                    batch_uttids, batch_wavs, batch_original_seq_lens, batch_original_audio_durs
                )
                
                uttid2asr_result.update(result)
                uttid2original_seq_len.update(seq_len)
                uttid2original_audio_dur.update(audio_dur)
                failed_batches += failed
                        
            except Exception as e:
                logger.error(f"Batch {batch_idx+1} 处理异常: {str(e)}", exc_info=True)
                failed_batches += 1
                continue

        logger.info(f"ASR推理完成，成功处理 {len(uttid2asr_result)} 个片段，失败 {failed_batches} 个片段")
        return uttid2asr_result

    def merge_results(self, audio_info_map: Dict[int, AudioInfo], uttid2asr_result: Dict):
        final_results = []
        validation_errors = []
        
        for audio_idx, audio_info in audio_info_map.items():
            try:
                segments = audio_info.segments
                sentences = []
                text_parts = []
                words = []
                processed_count = 0
                skipped_count = 0

                for seg in segments:
                    try:
                        if seg.audio_idx != audio_idx:
                            validation_errors.append(
                                f"audio_idx 不匹配！期望 {audio_idx}，实际 {seg.audio_idx}，片段 {seg.uttid}，跳过"
                            )
                            skipped_count += 1
                            continue
                        
                        if seg.uttid not in uttid2asr_result:
                            skipped_count += 1
                            continue

                        asr_res = uttid2asr_result[seg.uttid]

                        start_ms = round(seg.start_s * 1000)
                        end_ms = round(seg.end_s * 1000)

                        text = asr_res.get("text", "")
                        text_parts.append(text)

                        if self.config["asr"].get("return_timestamp", True):
                            sub_sentences = []
                            try:
                                sub_sentences = [{
                                    "start_ms": start_ms,
                                    "end_ms": end_ms,
                                    "text": text,
                                    "asr_confidence": asr_res.get("confidence", 0.0),
                                }]
                            except Exception as e:
                                logger.warning(f"构建句子失败，使用简化模式: uttid={seg.uttid}, error={str(e)}")
                                sub_sentences = [{
                                    "start_ms": start_ms,
                                    "end_ms": end_ms,
                                    "text": text,
                                    "asr_confidence": asr_res.get("confidence", 0.0),
                                }]
                            sentences.extend(sub_sentences)
                        else:
                            sentence = {
                                "start_ms": start_ms,
                                "end_ms": end_ms,
                                "text": text,
                                "asr_confidence": asr_res.get("confidence", 0.0),
                            }
                            sentences.append(sentence)

                        try:
                            if "timestamp" in asr_res:
                                for w, s, e in asr_res["timestamp"]:
                                    start_ms_word = max(0, round(s*1000 + start_ms))
                                    end_ms_word = max(start_ms_word, round(e*1000 + start_ms))
                                    words.append({
                                        "start_ms": start_ms_word,
                                        "end_ms": end_ms_word,
                                        "text": w
                                    })
                        except Exception as e:
                            logger.warning(f"提取词级时间戳失败: uttid={seg.uttid}, error={str(e)}")
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"处理片段失败，跳过: uttid={seg.uttid}, error={str(e)}", exc_info=True)
                        skipped_count += 1
                        continue
                
                if processed_count > 0 or skipped_count > 0:
                    logger.info(f"音频 {audio_info.audio_path}: 处理 {processed_count} 个片段，跳过 {skipped_count} 个片段")

                full_text = "".join(text_parts)

                result = {
                    "uttid": os.path.basename(audio_info.audio_path).split('.')[0],
                    "text": full_text,
                    "sentences": sentences,
                    "vad_segments_ms": [(round(seg.start_s*1000), round(seg.end_s*1000)) for seg in segments],
                    "dur_s": audio_info.dur_s,
                    "words": words,
                    "wav_path": audio_info.audio_path,
                    "original_file": audio_info.original_file,
                    "processed_segments": processed_count,
                    "skipped_segments": skipped_count
                }
                
                final_results.append(result)
                
            except Exception as e:
                logger.error(f"合并音频结果失败: audio_idx={audio_idx}, error={str(e)}", exc_info=True)
                continue
        
        if validation_errors:
            logger.warning(f"发现 {len(validation_errors)} 个验证错误:")
            for err in validation_errors[:10]:
                logger.warning(f"  - {err}")
            if len(validation_errors) > 10:
                logger.warning(f"  ... 还有 {len(validation_errors)-10} 个错误")
        
        return final_results

    def batch_process(self, audio_files: List[Dict], max_batch_dur_s=64, max_workers=8):
        if not audio_files:
            return []

        start_time = time.perf_counter()
        
        vad_start = time.perf_counter()
        all_segments, audio_info_map = self.batch_vad_process(audio_files, max_workers=max_workers)
        vad_duration = time.perf_counter() - vad_start
        
        if not all_segments:
            logger.warning("没有有效VAD片段")
            return []

        asr_start = time.perf_counter()
        uttid2asr = self.batch_asr_process(all_segments, audio_info_map, max_batch_dur_s=max_batch_dur_s)
        asr_duration = time.perf_counter() - asr_start

        merge_start = time.perf_counter()
        final_results = self.merge_results(audio_info_map, uttid2asr)
        merge_duration = time.perf_counter() - merge_start
        
        total_audio_dur = sum(info.dur_s for info in audio_info_map.values())
        total_processing_time = vad_duration + asr_duration + merge_duration
        rtf = total_processing_time / total_audio_dur if total_audio_dur > 0 else 0
        
        total_duration = time.perf_counter() - start_time

        logger.info(f"批量处理完成 - 总耗时: {total_duration:.2f}s "
                   f"(VAD: {vad_duration:.2f}s, ASR: {asr_duration:.2f}s, "
                   f"Merge: {merge_duration:.2f}s)")
        logger.info(f"性能统计 - 总音频时长: {total_audio_dur:.2f}s, 总处理用时: {total_processing_time:.2f}s, RTF: {rtf:.3f}x")

        self._clear_audio_cache()
        
        return final_results, {
            "total_audio_dur_s": total_audio_dur,
            "total_processing_s": total_processing_time,
            "rtf": rtf,
            "timing": {
                "vad_s": vad_duration,
                "asr_s": asr_duration,
                "punc_s": 0.0,
                "merge_s": merge_duration
            }
        }
