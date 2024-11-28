#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate mytorch

device=$1

dataset=$2

backbone=$3

log_dir="log/train_$(date '+%m%d-%H%M')"

mkdir ${log_dir}

filename="${dataset}_${backbone}_joint.txt"

python -u train.py --device $device --dataset $dataset --backbone $backbone --phase 'joint' 2>&1 | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_base.txt"

python -u train.py --device $device --dataset $dataset --backbone $backbone --phase 'base' 2>&1 | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_inc.txt"

python -u train.py --device $device --dataset $dataset --backbone $backbone --phase 'inc' 2>&1 | tee ${log_dir}/${filename}



