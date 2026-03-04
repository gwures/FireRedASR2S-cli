#!/bin/bash

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang)

export PATH=$PWD/fireredvad/bin:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

model_dir=$PWD/pretrained_models/FireRedVAD/VAD

if [ ! -f wav/wav.scp ]; then
    (cd wav && tar -zxvf wav.tar.gz)
fi

# Support several input format
wavs="--wav_path wav/BAC009S0764W0121.wav"
wavs="--wav_paths wav/BAC009S0764W0121.wav wav/IT0011W0001.wav"
wavs="--wav_scp wav/wav.scp"
wavs="--wav_dir wav/"

out=out/vad.txt
out_seg_dir=out/vad
mkdir -p $(dirname $out) $(dirname $out_seg_dir)
set -x

vad_config="--use_gpu 0 --smooth_window_size 5 --speech_threshold 0.5 --min_speech_frame 20 --max_speech_frame 2000 --min_silence_frame 10 --merge_silence_frame 50 --extend_speech_frame 5 --chunk_max_frame 30000"

CUDA_VISIBLE_DEVICES=0 \
vad.py --model_dir $model_dir $vad_config $wavs --output $out --write_textgrid 1 --save_segment_dir $out_seg_dir
