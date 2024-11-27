#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate mytorch

device=$1

dataset=$2

backbone=$3

filename="${backbone}_$(date '+%m%d-%H%M').txt"

python -u train.py --device $device --dataset $dataset --backbone $backbone --phase 'joint' 2>&1 | tee log/$filename

python -u train.py --device $device --dataset $dataset --backbone $backbone --phase 'base' 2>&1 | tee -a log/$filename

python -u train.py --device $device --dataset $dataset --backbone $backbone --phase 'inc' 2>&1 | tee -a log/$filename

