python3 test.py --model_name "roberta-base" \
                --num_labels 4 \
                --trained_checkpoint "PATH/TO/TRAINED/CHECKPOINT.ckpt" \
                --test_type "fixed_noise" \
                --test_noise 0.1 \
                --dataset_type "fused" \
                --wandb_key "YOUR WANDB KEY" \
                --santa_barbara_path "PATH/TO/SBCorpus/TRN" \
                --write_csv \
                --csv_save_path "PATH/TO/SAVE/CSV/FILE.csv" 

