#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate mytorch

filename="train_$(date '+%m%d-%H%M').txt"

# python -u train.py --phase 'base' 2>&1 | tee log/$filename

python -u train.py --mode 'finetune' 2>&1 | tee log/$filename
