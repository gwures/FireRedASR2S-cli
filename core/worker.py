import heapq
import logging
import os
import re
import time
from dataclasses import dataclass
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


def filter_timestamp_by_original_len(
    timestamp: List, original_seq_len: int, original_audio_dur: float
) -> List:
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


class MixedASRSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_paths = config["model_paths"]

        self.max_batch_dur_s = config.get("max_batch_dur_s", 64)
        self.use_bfd = config.get("use_bfd", False)

        logger.info(
            f"MixedASRSystem created (max_batch_dur_s={self.max_batch_dur_s}, use_bfd={self.use_bfd})"
        )

        self._load_vad()
        self._load_asr()

        logger.info("VAD and ASR preloaded!")

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

    @staticmethod
    def _load_audio(audio_path: str) -> np.ndarray:
        audio_data, _ = sf.read(audio_path, dtype="int16")
        return audio_data

    @staticmethod
    def _split_vad_to_batches_bfd(
        segments: List[VadSegment],
        max_batch_dur_s=64,
        max_bucket_num=5,
        max_ratio=2,
        short_seg_thresh=1.0,
    ):
        """
        BFD (Best-Fit Decreasing) 算法：动态聚类分桶 + 降序排序 + 最佳适应贪心 + 短片段填充
        :param segments: VAD片段列表（单个音频）
        :param max_batch_dur_s: ASR单batch最大总时长
        :param max_bucket_num: 最大桶数（避免过细分桶）
        :param max_ratio: 同桶最长/最短时长比
        :param short_seg_thresh: 短片段阈值（秒），≤该值的片段会做填充优化
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

        all_batches = []
        for bucket in buckets:
            if not bucket:
                continue
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
    def _split_vad_to_batches_wfd(
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
    def split_vad_to_batches(
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
            return MixedASRSystem._split_vad_to_batches_bfd(
                segments, max_batch_dur_s, max_bucket_num, max_ratio
            )
        else:
            return MixedASRSystem._split_vad_to_batches_wfd(
                segments, max_batch_dur_s, max_bucket_num, max_ratio
            )

    @staticmethod
    def _split_batch_by_duration(
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

    def _process_batch_with_adaptive_fallback(
        self,
        uttids: List[str],
        wavs: List[Tuple],
        seq_lens: List[int],
        audio_durs: List[float],
        current_max_dur: float = None,
        min_dur_s: float = 8.0,
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
            batch_asr_results = [
                a for a in batch_asr_results if not _BLANK_SIL_PATTERN.search(a["text"])
            ]

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
                                    res["timestamp"] = filter_timestamp_by_original_len(
                                        res["timestamp"], seq_lens[i], audio_durs[i]
                                    )
                                uttid2result[uttids[i]] = res
                                uttid2seq_len[uttids[i]] = seq_lens[i]
                                uttid2audio_dur[uttids[i]] = audio_durs[i]
                        except Exception as inner_e:
                            logger.error(
                                f"Failed to process single segment {uttids[i]}: {inner_e}"
                            )
                            failed_count += 1
                    return uttid2result, uttid2seq_len, uttid2audio_dur, failed_count

                target_dur = current_max_dur or (total_dur // 2)
                target_dur = max(target_dur, min_dur_s)
                logger.info(f"Retrying with target_dur={target_dur:.1f}s")

                sub_batch_indices = self._split_batch_by_duration(
                    audio_durs, target_dur
                )

                for sub_indices in sub_batch_indices:
                    sub_uttids = [uttids[i] for i in sub_indices]
                    sub_wavs = [wavs[i] for i in sub_indices]
                    sub_seq_lens = [seq_lens[i] for i in sub_indices]
                    sub_audio_durs = [audio_durs[i] for i in sub_indices]

                    sub_result, sub_seq, sub_dur, sub_failed = (
                        self._process_batch_with_adaptive_fallback(
                            sub_uttids,
                            sub_wavs,
                            sub_seq_lens,
                            sub_audio_durs,
                            current_max_dur=target_dur,
                            min_dur_s=min_dur_s,
                        )
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

        try:
            logger.info(f"[{audio_idx}] 开始处理音频: {audio_path}")

            audio_data = self._load_audio(audio_path)
            dur_s = audio_data.shape[0] / SAMPLE_RATE
            logger.debug(f"[{audio_idx}] 音频加载完成，时长: {dur_s:.2f}s")

            vad_result, _ = self.vad.detect((SAMPLE_RATE, audio_data))

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

            batches = self.split_vad_to_batches(
                audio_segments, max_batch_dur_s=max_batch_dur_s, use_bfd=self.use_bfd
            )
            logger.info(f"[{audio_idx}] 分批完成，共 {len(batches)} 个 batch")

            uttid2asr_result = {}
            failed_batches = 0

            logger.info(f"[{audio_idx}] 开始 ASR 推理，共 {len(batches)} 个 batch")

            for batch_idx, seg_indices in enumerate(batches):
                try:
                    batch_segs = [audio_segments[i] for i in seg_indices]

                    batch_uttids = []
                    batch_wavs = []
                    batch_seq_lens = []
                    batch_audio_durs = []

                    for seg in batch_segs:
                        start_sample = round(seg.start_s * SAMPLE_RATE)
                        end_sample = round(seg.end_s * SAMPLE_RATE)
                        wav_segment = audio_data[start_sample:end_sample]

                        original_audio_dur = seg.end_s - seg.start_s
                        original_audio_samples = len(wav_segment)
                        original_seq_len = calc_original_seq_len(original_audio_samples)

                        batch_uttids.append(seg.uttid)
                        batch_wavs.append((SAMPLE_RATE, wav_segment))
                        batch_seq_lens.append(original_seq_len)
                        batch_audio_durs.append(original_audio_dur)

                    logger.debug(
                        f"[{audio_idx}] ASR batch {batch_idx+1}/{len(batches)}: {len(batch_wavs)} segments"
                    )

                    result, seq_len, audio_dur, failed = (
                        self._process_batch_with_adaptive_fallback(
                            batch_uttids, batch_wavs, batch_seq_lens, batch_audio_durs
                        )
                    )

                    uttid2asr_result.update(result)
                    failed_batches += failed

                    logger.info(
                        f"[{audio_idx}] ASR batch {batch_idx+1}/{len(batches)} 完成，{len(batch_wavs)} 个片段"
                    )

                except Exception as e:
                    logger.error(
                        f"[{audio_idx}] Batch {batch_idx+1} 处理异常: {str(e)}",
                        exc_info=True,
                    )
                    failed_batches += 1

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

    def batch_process(self, audio_files: List[Dict], max_batch_dur_s=64):
        """
        逐个处理音频文件

        Args:
            audio_files: 音频文件信息列表
            max_batch_dur_s: ASR batch 最大时长

        Returns:
            (final_results, stats): 处理结果列表和统计信息
        """
        if not audio_files:
            return [], {}

        process_results: List[AudioProcessResult] = []

        for audio_idx, audio_info in enumerate(audio_files):
            result = self._process_single_audio(
                audio_idx=audio_idx,
                audio_info=audio_info,
                max_batch_dur_s=max_batch_dur_s,
            )
            process_results.append(result)

        final_results = []
        rtfs = []
        total_audio_dur = 0.0
        total_processing_time = 0.0
        success_count = 0
        error_count = 0

        for pr in process_results:
            if pr.success and pr.result:
                final_results.append(pr.result)
                rtfs.append(pr.rtf)
                total_audio_dur += pr.result["dur_s"]
                total_processing_time += pr.processing_time_s
                success_count += 1
            else:
                error_count += 1
                logger.warning(f"音频 {pr.audio_path} 处理失败: {pr.error_msg}")

        avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0
        overall_rtf = (
            total_processing_time / total_audio_dur if total_audio_dur > 0 else 0.0
        )

        logger.info(f"批量处理完成 - 成功: {success_count}, 失败: {error_count}")
        logger.info(
            f"总音频时长: {total_audio_dur:.2f}s, 总处理用时: {total_processing_time:.2f}s"
        )
        logger.info(f"平均 RTF: {avg_rtf:.3f}x, 整体 RTF: {overall_rtf:.3f}x")

        stats = {
            "total_audio_dur_s": total_audio_dur,
            "total_processing_s": total_processing_time,
            "avg_rtf": avg_rtf,
            "overall_rtf": overall_rtf,
            "success_count": success_count,
            "error_count": error_count,
            "per_audio_rtfs": rtfs,
        }

        return final_results, stats
