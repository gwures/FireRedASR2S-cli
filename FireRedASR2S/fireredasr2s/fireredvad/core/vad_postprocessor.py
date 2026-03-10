# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang)

import enum
import logging
from typing import List, Tuple

import numpy as np

from .constants import FRAME_LENGTH_S, FRAME_SHIFT_S

logger = logging.getLogger(__name__)


@enum.unique
class VadState(enum.Enum):
    SILENCE = 0
    POSSIBLE_SPEECH = 1
    SPEECH = 2
    POSSIBLE_SILENCE = 3


class VadPostprocessor:
    def __init__(
        self,
        smooth_window_size,
        prob_threshold,
        min_speech_frame,
        max_speech_frame,
        min_silence_frame,
        merge_silence_frame,
        extend_speech_frame,
    ):
        self.smooth_window_size = max(1, smooth_window_size)
        self.prob_threshold = prob_threshold
        self.min_speech_frame = min_speech_frame
        self.max_speech_frame = max_speech_frame
        self.min_silence_frame = min_silence_frame
        self.merge_silence_frame = merge_silence_frame
        self.extend_speech_frame = extend_speech_frame

    def process(self, raw_probs):
        if raw_probs is None or len(raw_probs) == 0:
            return []
        return self._process_optimized(raw_probs)

    def _process_optimized(self, raw_probs: np.ndarray) -> List[int]:
        n = len(raw_probs)
        if n == 0:
            return []

        probs_np = np.asarray(raw_probs, dtype=np.float32)
        smoothed_probs = self._smooth_prob(probs_np)
        binary_preds = self._apply_threshold_np(smoothed_probs)

        decisions = np.zeros(n, dtype=np.int8)
        state = VadState.SILENCE
        speech_start = -1
        silence_start = -1

        for t, is_speech in enumerate(binary_preds):
            if state == VadState.SILENCE:
                if is_speech:
                    state = VadState.POSSIBLE_SPEECH
                    speech_start = t

            elif state == VadState.POSSIBLE_SPEECH:
                if is_speech:
                    if t - speech_start >= self.min_speech_frame:
                        state = VadState.SPEECH
                        decisions[speech_start:t] = 1
                else:
                    state = VadState.SILENCE
                    speech_start = -1

            elif state == VadState.SPEECH:
                if not is_speech:
                    state = VadState.POSSIBLE_SILENCE
                    silence_start = t

            elif state == VadState.POSSIBLE_SILENCE:
                if not is_speech:
                    if t - silence_start >= self.min_silence_frame:
                        state = VadState.SILENCE
                        speech_start = -1
                else:
                    state = VadState.SPEECH
                    silence_start = -1

            if state == VadState.SPEECH or state == VadState.POSSIBLE_SILENCE:
                decisions[t] = 1
            else:
                decisions[t] = 0

        speech_transitions = np.diff(decisions, prepend=0)
        for t in np.where(speech_transitions == 1)[0]:
            start = max(0, t - self.smooth_window_size)
            decisions[start:t] = 1

        if self.merge_silence_frame > 0:
            silence_start = None
            for t in range(1, n):
                if (
                    decisions[t - 1] == 1
                    and decisions[t] == 0
                    and silence_start is None
                ):
                    silence_start = t
                elif (
                    decisions[t - 1] == 0
                    and decisions[t] == 1
                    and silence_start is not None
                ):
                    silence_frame = t - silence_start
                    if silence_frame < self.merge_silence_frame:
                        decisions[silence_start:t] = 1
                    silence_start = None

        if self.extend_speech_frame > 0:
            decisions = self._extend_speech_segments_fast(decisions)

        if self.max_speech_frame > 0:
            speech_segments = self._extract_speech_segments_np(decisions)
            for start_frame, end_frame in speech_segments:
                dur_frames = end_frame - start_frame
                if dur_frames > self.max_speech_frame:
                    segment_probs = probs_np[start_frame:end_frame]
                    split_points = self._find_split_points(segment_probs)
                    for split_point in split_points:
                        split_frame = start_frame + split_point
                        if 0 <= split_frame < n:
                            decisions[split_frame] = 0

        return decisions.tolist()

    def _extract_speech_segments(self, decisions: List[int]) -> List[Tuple[int, int]]:
        segments = []
        speech_start = None
        for t, decision in enumerate(decisions):
            if decision == 1 and speech_start is None:
                speech_start = t
            elif decision == 0 and speech_start is not None:
                segments.append((speech_start, t))
                speech_start = None
        if speech_start is not None:
            segments.append((speech_start, len(decisions)))
        return segments

    def _extend_speech_segments_fast(self, decisions):
        decisions_np = np.asarray(decisions, dtype=np.int8)
        kernel = np.ones(2 * self.extend_speech_frame + 1, dtype=np.int8)
        extended = np.convolve(decisions_np, kernel, mode="same")
        return (extended > 0).astype(np.int8)

    def decision_to_segment(self, decisions, wav_dur=None):
        segments = []
        speech_start = None
        for t, decision in enumerate(decisions):
            if decision == 1 and speech_start is None:
                speech_start = t
            elif decision == 0 and speech_start is not None:
                if (t - speech_start) < self.min_speech_frame:
                    logger.warning(
                        "Unexpected short speech segment, check vad_postprocessor.py"
                    )
                segments.append((speech_start * FRAME_SHIFT_S, t * FRAME_SHIFT_S))
                speech_start = None
        if speech_start is not None:
            t = len(decisions) - 1
            if (t - speech_start) < self.min_speech_frame:
                logger.warning(
                    "Unexpected short speech segment, check vad_postprocessor.py"
                )
            end_time = len(decisions) * FRAME_SHIFT_S + FRAME_LENGTH_S
            if wav_dur is not None:
                end_time = min(end_time, wav_dur)
            segments.append((speech_start * FRAME_SHIFT_S, end_time))
        segments = [(round(s, 3), round(e, 3)) for s, e in segments]
        return segments

    def _smooth_prob(self, probs):
        if self.smooth_window_size <= 1:
            return np.asarray(probs, dtype=np.float32)
        probs_np = np.array(probs, dtype=np.float32)
        kernel = (
            np.ones(self.smooth_window_size, dtype=np.float32) / self.smooth_window_size
        )
        smoothed = np.convolve(probs_np, kernel, mode="full")[: len(probs)]
        for i in range(min(self.smooth_window_size - 1, len(probs))):
            smoothed[i] = np.mean(probs_np[: i + 1])
        return smoothed

    def _apply_threshold(self, probs):
        probs_np = np.asarray(probs, dtype=np.float32)
        return (probs_np >= self.prob_threshold).astype(int).tolist()

    def _apply_threshold_np(self, probs: np.ndarray) -> np.ndarray:
        return probs >= self.prob_threshold

    def _extract_speech_segments_np(
        self, decisions: np.ndarray
    ) -> List[Tuple[int, int]]:
        segments = []
        speech_start = None
        for t in range(len(decisions)):
            if decisions[t] == 1 and speech_start is None:
                speech_start = t
            elif decisions[t] == 0 and speech_start is not None:
                segments.append((speech_start, t))
                speech_start = None
        if speech_start is not None:
            segments.append((speech_start, len(decisions)))
        return segments

    def _find_split_points(self, probs):
        split_points = []
        length = len(probs)
        start = 0
        while start < length:
            if (length - start) <= self.max_speech_frame:
                break
            window_start = int(start + self.max_speech_frame / 2)
            window_end = int(start + self.max_speech_frame)
            window_probs = probs[window_start:window_end]

            min_index = window_start + np.argmin(window_probs)
            split_points.append(min_index)

            start = min_index + 1
        return split_points
