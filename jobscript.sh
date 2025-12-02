#!/bin/bash
#BSUB -J train_byte
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -W 12:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o train%J.out
#BSUB -e train%J.err

module load cuda/11.8

source .venv/bin/activate

python train.py