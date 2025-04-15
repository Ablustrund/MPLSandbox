#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
export HF_ENDPOINT="https://hf-mirror.com"


model=TinyLlama/TinyLlama-1.1B-Chat-v0.1
eval_steps=50
beta=0.05
rollout=1

job_name=mplsandbox_for_ppo${eval_steps}_beta005_rollout${rollout}_0508_debug
tmp_path=./tmp/$job_name
# if [ -d $tmp_path ]; then
#     rm -r $tmp_path/*
# fi
if [ ! -d "$tmp_path" ]; then
    mkdir -p "$tmp_path"
else
    if [ "$(ls -A "$tmp_path")" ]; then
        rm -r "$tmp_path"/*
    fi
fi
data_path=./data

echo "please see log: ./log/$job_name.log"

CUDA_VISIBLE_DEVICES=0 \
accelerate launch --config_file config.yaml --num_processes 1 train_ppo.py \
--hf_model_name $model \
--init_actor $model \
--init_reward $model \
--model_file outputs/ppo/$job_name \
--train_steps 1000 --warmup_steps $eval_steps --save_freq $eval_steps \
--batch_size $rollout --rollout_batch_size $rollout --n_rollouts $rollout --n_candidates 1 \
--context_truncate 2048 --max_ts 1024 \
--data_path $data_path \
--validation_metric rewards \
--scheduler invsqrt --lr 1e-6 --eps 1e-6 --beta2 0.95 --clip_reward 0 --beta $beta  \
--gradient_checkpoint \
--belle_style_prompt --delimiter '</s>' \
--inference nucleus --topp 0.95 --temperature 0.2 \
--tensorboard_logdir ./tensorboard_log/ppo/$job_name &> ./log/$job_name.log