# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

import math
import os
from typing import List

import kaldiio
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torchaudio.compliance.kaldi as kaldi


class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(
            num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
        )

    def __call__(self, wav_paths, wav_uttids):
        feats = []
        durs = []
        return_wav_paths = []
        return_wav_uttids = []

        wav_datas = []
        if isinstance(wav_paths[0], str):
            for wav_path in wav_paths:
                sample_rate, wav_np = kaldiio.load_mat(wav_path)
                if wav_np.dtype != np.float32:
                    wav_np = wav_np.astype(np.float32)
                wav_datas.append([sample_rate, wav_np])
        else:
            wav_datas = wav_paths

        for (sample_rate, wav_np), path, uttid in zip(wav_datas, wav_paths, wav_uttids):
            dur = wav_np.shape[0] / sample_rate
            fbank = self.fbank((sample_rate, wav_np))
            if fbank.shape[0] < 1:
                continue
            if self.cmvn is not None:
                fbank = self.cmvn(fbank)
            feats.append(fbank)
            durs.append(dur)
            return_wav_paths.append(path)
            return_wav_uttids.append(uttid)
        if len(feats) > 0:
            lengths = torch.tensor([feat.size(0) for feat in feats]).long()
            feats_pad = self.pad_feat(feats, 0.0)
        else:
            lengths, feats_pad = None, None
        return feats_pad, lengths, durs, return_wav_paths, return_wav_uttids

    def extract_from_cached_fbank(self, cached_fbank, segment_infos):
        """
        从缓存的 fbank 中提取片段特征，避免重复计算

        Args:
            cached_fbank: VAD 阶段缓存的原始 fbank tensor, shape [T, 80]
            segment_infos: List of (start_frame, end_frame, uttid, dur) tuples

        Returns:
            feats_pad, lengths, durs, uttids
        """
        feats = []
        durs = []
        uttids = []

        for start_frame, end_frame, uttid, dur in segment_infos:
            seg_fbank = cached_fbank[start_frame:end_frame].clone()
            if seg_fbank.shape[0] < 1:
                continue
            if self.cmvn is not None:
                seg_fbank = self.cmvn(seg_fbank)
            feats.append(seg_fbank)
            durs.append(dur)
            uttids.append(uttid)

        if len(feats) > 0:
            lengths = torch.tensor([feat.size(0) for feat in feats]).long()
            feats_pad = self.pad_feat(feats, 0.0)
        else:
            lengths, feats_pad = None, None
        return feats_pad, lengths, durs, uttids

    def pad_feat(self, xs: List[torch.Tensor], pad_value: float) -> torch.Tensor:
        return rnn_utils.pad_sequence(xs, batch_first=True, padding_value=pad_value)


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = self.read_kaldi_cmvn(
            kaldi_cmvn_file
        )

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean * mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return (
            dim,
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(inverse_std_variences, dtype=torch.float32),
        )


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0):
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.snip_edges = True

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
            if wav_np.dtype != np.float32:
                wav_np = wav_np.astype(np.float32)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)
        feat = kaldi.fbank(
            wav_tensor,
            num_mel_bins=self.num_mel_bins,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            dither=dither,
            snip_edges=self.snip_edges,
            sample_frequency=16000,
        )
        return feat
