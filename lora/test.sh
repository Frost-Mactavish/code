#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate mytorch

filename="test_$(date '+%m%d-%H%M').txt"

for file in ./checkpoints/*.pth; do
    if [ -f "$file" ]; then
      file=$(basename $file)
      python -u dior_test.py --phase 'joint' --test_mode 'map' --weight_file $file 2>&1 | tee -a log/${filename}
    fi
done