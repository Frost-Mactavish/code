#!/bin/bash
nvcc ViT.cu -o ViT-CUDA -lcublas
python ViT.py > result.txt
./ViT-CUDA >> result.txt
