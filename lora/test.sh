#!/bin/bash

source /home/freddy/Software/miniforge/etc/profile.d/conda.sh

conda activate mytorch

filename="test_$(date '+%m%d-%H%M').txt"

#for file in ./checkpoints/*DIOR-base*.pth; do
#    if [ -f "$file" ]; then
#      file=$(basename $file)
#      python -u dior_test.py --phase 'base' --test_mode 'test' --weight_file $file 2>&1 | tee log/$filename
#    fi
#done
#
#for file in ./checkpoints/*DIOR-inc*.pth; do
#    if [ -f "$file" ]; then
#      file=$(basename $file)
#      python -u dior_test.py --phase 'inc' --test_mode 'test' --weight_file $file 2>&1 | tee -a log/$filename
#    fi
#done

for file in ./checkpoints/*.pth; do
    if [ -f "$file" ]; then
      file=$(basename $file)
      python -u dior_test.py --phase 'joint' --test_mode 'map' --weight_file $file 2>&1 | tee -a log/$filename
    fi
done