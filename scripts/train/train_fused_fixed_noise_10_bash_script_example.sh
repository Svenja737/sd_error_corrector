python3 train.py --model_name "roberta-base" \
                 --num_labels 4 \
                 --dataset_type "fused" \
                 --training_mode "fixed_noise" \
                 --wandb_key "YOUR WANDB KEY" \
                 --label_noise 0.1 \
                 --token_noise "False" \
                 --santa_barbara_path "PATH/TO/SBCorpus/TRN"


