#!/bin/bash

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang)

export PATH=$PWD/fireredvad/bin:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

model_dir=$PWD/pretrained_models/FireRedVAD/AED

if [ ! -f wav4aed/1.wav ]; then
    (cd wav4aed && tar -zxvf wav.tar.gz)
fi

# Support several input format
wavs="--wav_path wav4aed/1.wav"
wavs="--wav_paths wav4aed/1.wav wav4aed/2.wav"
wavs="--wav_scp wav4aed/wav.scp"
wavs="--wav_dir wav4aed/"

out=out/aed.txt
out_seg_dir=out/aed
mkdir -p $(dirname $out) $(dirname $out_seg_dir)
set -x

aed_config="--use_gpu 0 --smooth_window_size 5 --speech_threshold 0.4 --singing_threshold 0.5 --music_threshold 0.5 --min_event_frame 20 --max_event_frame 2000 --min_silence_frame 20 --merge_silence_frame 0 --extend_speech_frame 0 --chunk_max_frame 30000"

CUDA_VISIBLE_DEVICES=0 \
aed.py --model_dir $model_dir $aed_config $wavs --output $out --write_textgrid 1 --save_segment_dir $out_seg_dir
