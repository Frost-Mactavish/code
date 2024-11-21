#! /bin/bash

source /home/freddy/Software/miniconda/etc/profile.d/conda.sh

conda activate mytorch

filename="train_$(date '+%m%d-%H%M').txt"

python -u train.py > ./log/$filename
