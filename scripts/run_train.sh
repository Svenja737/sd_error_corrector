#!/usr/bin/env bash

#SBATCH --job-name=sd_correction_model
#SBATCH --output=sd_correction_model.txt
#SBATCH --ntasks=1

#SBATCH --mem=32000
#SBATCH --mail-user=filthaut@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 train_test_model.py