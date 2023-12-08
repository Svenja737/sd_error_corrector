from sdec_pipeline import SDECPipeline
import os
import wandb
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta-base", help="Backbone model for extracting word embeddings (default: roberta-base).")
    parser.add_argument("--num_labels", type=int, help="Number of labels, equal to max speakers in one sample in data.")
    parser.add_argument("--dataset_type", choices=["switchboard", "santa_barbara", "fused"], help="Dataset variant, either Switchboard, Santa Barbara or both (i.e. fused).")
    parser.add_argument("--training_mode", choices=["no_noise", "fixed_noise", "scheduled_noise", "overlap_noise"], help="Training mode setting noise in concatenated labels.")
    parser.add_argument("--token_noise", choices=["True", "False"], help="Inject noise into tokens as well as labels.")
    parser.add_argument("--wandb_key", help="Authentification key for wandb logger.")
    parser.add_argument("--label_noise", type=float, default=0.0, help="Amount of label noise in training mode 'fixed_noise.")
    parser.add_argument("--santa_barbara_path", default="", help="Path to downloaded SB dataset. Download at https://www.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7/files/sitefiles/research/SBC/SBCorpus.zip")
    args = parser.parse_args()

    wandb.login(anonymous="allow", key=args.wandb_key)
    sdec = SDECPipeline()
    sdec.train_model(args.model_name, args.num_labels, args.dataset_type, args.training_mode, args.token_noise, args.label_noise, args.santa_barbara_path)

if __name__ == "__main__":
    main()