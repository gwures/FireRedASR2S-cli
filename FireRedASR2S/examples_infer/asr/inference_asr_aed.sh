#!/bin/bash

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

export PATH=$PWD/fireredasr2/:$PWD/fireredasr2/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH

if [ ! -f wav/wav.scp ]; then
    (cd wav && tar -zxvf wav.tar.gz)
fi

# model_dir includes model.pth.tar, cmvn.ark, dict.txt
model_dir=$PWD/pretrained_models/FireRedASR2-AED

# Support several input format
wavs="--wav_path wav/BAC009S0764W0121.wav"
wavs="--wav_paths wav/BAC009S0764W0121.wav wav/IT0011W0001.wav wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav wav/TEST_MEETING_T0000000001_S00000.wav"
wavs="--wav_dir wav/"
wavs="--wav_scp wav/wav.scp"
ref="wav/text"
#wavs="--wav_dir wav4aed"

out="out/aed-l-asr.txt"

decode_args="
--use_gpu 1 --use_half 0
--sort_wav_by_dur 1 --batch_size 64 --beam_size 3 --nbest 1
--decode_max_len 300 --softmax_smoothing 1.25 --aed_length_penalty 0.6
--eos_penalty 1.0
--elm_weight 0.0
--return_timestamp 1 --write_textgrid 1
"

mkdir -p $(dirname $out)
set -x


CUDA_VISIBLE_DEVICES=0 \
speech2text.py --asr_type "aed" --model_dir $model_dir $decode_args $wavs --output $out


wer.py --print_sentence_wer 1 --do_tn 1 --rm_special 0 --ref $ref --hyp $out > $out.wer 2>&1
tail -n8 $out.wer
