#!/bin/bash
#SBATCH --job-name=bpe_owt
#SBATCH --partition=shared
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=bpe_%j.log

cd ~/cs336-assignment1
python -m cs336_basics.training_tokenizer_owt
