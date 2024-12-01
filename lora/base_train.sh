#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate torch

dataset=$1

backbone=$2

log_dir="log/base_$(date '+%m%d-%H%M')"

mkdir ${log_dir}

filename="${dataset}_${backbone}_joint.txt"

python -u train.py --dataset ${dataset} --backbone ${backbone} --phase 'joint' 2>&1 \
--partial None --resume None | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_base.txt"

python -u train.py --dataset ${dataset} --backbone ${backbone} --phase 'base' 2>&1 \
--partial None --resume None | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_inc.txt"

python -u train.py --dataset ${dataset} --backbone ${backbone} --phase 'inc' 2>&1 \
--partial None --resume None | tee ${log_dir}/${filename}