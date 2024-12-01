#!/bin/bash

source /data/anaconda/etc/profile.d/conda.sh

conda activate torch

device=$1

dataset=$2

backbone=$3

weight=$4

log_dir="log/finetune_${backbone}_$(date '+%m%d-%H%M')"

mkdir ${log_dir}

filename="${dataset}_${backbone}_inc_full.txt"

python -u train.py --device ${device} --dataset ${dataset} --backbone ${backbone} --phase 'inc' \
--resume ${weight} 2>&1 | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_inc_1.txt"

python -u train.py --device ${device} --dataset ${dataset} --backbone ${backbone} --phase 'inc' \
--partial 1 --resume ${weight} 2>&1 | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_inc_2.txt"

python -u train.py --device ${device} --dataset ${dataset} --backbone ${backbone} --phase 'inc' \
--partial 2 --resume ${weight} 2>&1 | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_inc_3.txt"

python -u train.py --device ${device} --dataset ${dataset} --backbone ${backbone} --phase 'inc' \
--partial 3 --resume ${weight} 2>&1 | tee ${log_dir}/${filename}

filename="${dataset}_${backbone}_inc_4.txt"

python -u train.py --device ${device} --dataset ${dataset} --backbone ${backbone} --phase 'inc' \
--partial 4 --resume ${weight} 2>&1 | tee ${log_dir}/${filename}