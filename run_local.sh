#!/bin/bash

exp_name=$1

export WANDB_API_KEY=xxx

if [ $# -eq 0 ];
then
    echo "Usage: ./run.sh  exp_name"
    exit
fi

export CUDA_VISIBLE_DEVICES=2

NOW=$(date +"%Y%m%d%H%M")

log_path=logs/$NOW

mkdir -p $log_path

echo "create log dir $log_path"


python train.py \
    -exp_id $exp_name \
    -datasets wikisql_label_uuid\
    -threads 8 \
    -plm bert-large \
    -model UniG \
    -bs 8 \
    -ls 0.05 \
    -out_dir ./output/ > $log_path/log.txt
