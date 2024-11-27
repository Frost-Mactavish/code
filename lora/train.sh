#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate mytorch

filename="train_$(date '+%m%d-%H%M').txt"

python -u train.py --dataset 'DIOR' --backbone 'resnet101' --phase 'joint' 2>&1 | tee log/$filename

python -u train.py --dataset 'DIOR' --backbone 'resnet101' --phase 'base' 2>&1 | tee -a log/$filename

python -u train.py --dataset 'DIOR' --backbone 'resnet101' --phase 'inc' 2>&1 | tee -a log/$filename