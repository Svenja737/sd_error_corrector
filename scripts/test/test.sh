#!/usr/bin/env bash

#SBATCH --job-name=sdec_test
#SBATCH --output=sdec_test.txt
#SBATCH --ntasks=1

#SBATCH --mem=32000
#SBATCH --mail-user=filthaut@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 test.py "roberta-base" 11 "/home/sfilthaut/sdec_revamped/sdec_revamped/models/epoch=6-step=9030.ckpt" "fused" "fa9a8ca94055c30f4a506593d5ed4d3a3cad70f2" --santa_barbara_path "/home/sfilthaut/sdec_revamped/SBCorpus/TRN" --write_csv