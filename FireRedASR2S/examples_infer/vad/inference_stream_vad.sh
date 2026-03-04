#!/bin/bash

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang)

export PATH=$PWD/fireredvad/bin/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

model_dir=$PWD/pretrained_models/FireRedVAD/Stream-VAD

if [ ! -f wav/wav.scp ]; then
    (cd wav && tar -zxvf wav.tar.gz)
fi

# Support several input format
wavs="--wav_path wav/BAC009S0764W0121.wav"
wavs="--wav_paths wav/BAC009S0764W0121.wav wav/IT0011W0001.wav"
wavs="--wav_scp wav/wav.scp"
wavs="--wav_dir wav/"

out="out/stream_vad.txt"
save_segment_dir=out/stream_vad
mkdir -p $(dirname $out) $save_segment_dir
set -x

vad_config="--use_gpu 0 --smooth_window_size 5 --speech_threshold 0.3 --pad_start_frame 5 --min_speech_frame 8 --max_speech_frame 2000 --min_silence_frame 20 --chunk_max_frame 30000"

CUDA_VISIBLE_DEVICES=0 \
stream_vad.py --model_dir "$model_dir" $vad_config $wavs --output "$out" --write_textgrid 1 --save_segment_dir $save_segment_dir
