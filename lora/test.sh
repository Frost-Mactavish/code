#!/bin/bash

source /data/anaconda/etc/profile.d/conda.sh

conda activate torch

device=$1

dataset=$2

filename="test_$(date '+%m%d-%H%M').txt"

for file in ./checkpoints/finetune/*.pth; do
    if [ -f "$file" ]; then
      file=$(basename $file)
      python -u dior_test.py --device ${device} --dataset ${dataset}  --test_mode 'map' --filename finetune/${file} 2>&1 | tee -a log/${filename}
    fi
done