#!/bin/bash

random_seed=42

python3 -m src.main \
    --gpu=1,2 \
    --save_dir=/deepall_mixup_Utrecht_testPerSubject_300epoch-rs$((random_seed)) \
    --normal_aug \
    --wandb=wmh_seg_dg \
    --random_seed=$random_seed \
    --max_num_epochs=300 \
    --batch_size=30 \
    --T1 \
    --single_target=Utrecht \
    --mixup_rate=0.7

python3 -m src.main \
    --gpu=1,2 \
    --save_dir=/deepall_mixup_GE3T_testPerSubject_300epoch-rs$((random_seed)) \
    --normal_aug \
    --wandb=wmh_seg_dg \
    --random_seed=$random_seed \
    --max_num_epochs=300 \
    --batch_size=30 \
    --T1 \
    --single_target=GE3T \
    --mixup_rate=0.7

python3 -m src.main \
    --gpu=1,2 \
    --save_dir=/deepall_mixup_Singapore_testPerSubject_300epoch-rs$((random_seed)) \
    --normal_aug \
    --wandb=wmh_seg_dg \
    --random_seed=$random_seed \
    --max_num_epochs=300 \
    --batch_size=30 \
    --T1 \
    --single_target=Singapore \
    --mixup_rate=0.7