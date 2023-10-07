#!/usr/bin/env bash

#SBATCH --job-name=sdec_00
#SBATCH --output=sdec_00.txt
#SBATCH --ntasks=1

#SBATCH --mem=32000
#SBATCH --mail-user=filthaut@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 train.py "roberta-base" "no_noise" 11 0.0 "fused" "fa9a8ca94055c30f4a506593d5ed4d3a3cad70f2" --santa_barbara_path "/home/sfilthaut/sdec_revamped/SBCorpus/TRN" 