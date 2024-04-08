#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python my_train.py \
    --dataset_name 'cifar100' \
    --sup_weight 0.35 \
    --enable_DInfoNCE False \
    --top_k 250 \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 4 \
    --exp_name cifar100_npc
