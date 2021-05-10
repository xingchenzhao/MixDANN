#!/bin/bash

random_seed=42

python3 -m src.main \
    --gpu=1,2 \
    --save_dir=/dann_T1_testPerSubject_utrecht_dw0.1dd3_d1-rs$((random_seed)) \
    --normal_aug \
    --wandb=wmh_seg_dg \
    --random_seed=$random_seed \
    --domain_adversary \
    --domain_adversary_weight=0.1 \
    --suppression_decay=3 \
    --early_adversary_suppression \
    --max_num_epochs=300 \
    --T1 \
    --single_target=Utrecht

python3 -m src.main \
    --gpu=1,2 \
    --save_dir=/dann_T1_testPerSubject_GE3T_dw0.1dd3_d1-rs$((random_seed)) \
    --normal_aug \
    --wandb=wmh_seg_dg \
    --random_seed=$random_seed \
    --domain_adversary \
    --domain_adversary_weight=0.1 \
    --suppression_decay=3 \
    --early_adversary_suppression \
    --max_num_epochs=300 \
    --T1 \
    --single_target=GE3T


python3 -m src.main \
    --gpu=1,2 \
    --save_dir=/dann_T1_testPerSubject_Singapore_dw0.1dd3_d1-rs$((random_seed)) \
    --normal_aug \
    --wandb=wmh_seg_dg \
    --random_seed=$random_seed \
    --domain_adversary \
    --domain_adversary_weight=0.1 \
    --suppression_decay=3 \
    --early_adversary_suppression \
    --max_num_epochs=300 \
    --T1 \
    --single_target=Singapore