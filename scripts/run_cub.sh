#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python my_train.py \
    --dataset_name 'cub' \
    --top_k 10 \
    --memax_weight 2 \
    --sup_weight 0.35 \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 4 \
    --use_ssb_splits \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --exp_name cub_npc
