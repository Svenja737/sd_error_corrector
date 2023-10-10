#!/usr/bin/env bash

#SBATCH --job-name=sdec_test_fused_nonoise
#SBATCH --output=sdec_test_fused_nonoise.txt
#SBATCH --ntasks=1

#SBATCH --mem=32000
#SBATCH --mail-user=filthaut@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --cpus-per-task=4
#SBATCH --qos=batch
#SBATCH --gres=gpu

python3 test.py "roberta-base" 10 "/home/students/filthaut/sdec_revamped/models/fused_no_noise.ckpt" "fused" "fa9a8ca94055c30f4a506593d5ed4d3a3cad70f2" --testing_mode "no_noise" --santa_barbara_path "/home/students/filthaut/sdec_revamped/data/SBCorpus/TRN" --write_csv --csv_save_path "/home/students/filthaut/sdec_revamped/results/test_fused_no_noise.csv"
