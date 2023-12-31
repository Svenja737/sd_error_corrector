from sdec_pipeline import SDECPipeline
import os
import wandb
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_labels", type=int, help="Number of labels, equal to max speakers in one sample in data.")
    parser.add_argument("--trained_checkpoint", help="Trained SDEC model checkpoint.")
    parser.add_argument("--dataset_type", choices=["switchboard", "santa_barbara", "fused"], help="Dataset variant, either Switchboard, Santa Barbara or both (i.e. fused).")
    parser.add_argument("--test_type", choices=["fixed_noise", "overlap_noise", "overlap_token_noise", "None"])
    parser.add_argument("--label_noise", type=float)
    parser.add_argument("--wandb_key", help="Authentification key for wandb logger.")
    parser.add_argument("--santa_barbara_path", help="Path to downloaded SB dataset. Download at https://www.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7/files/sitefiles/research/SBC/SBCorpus.zip")
    parser.add_argument("--write_csv", action="store_true", help="Save model information during testing.")
    parser.add_argument("--csv_save_path", default="results", help="Location of the model information csv file.")
    args = parser.parse_args()

    wandb.login(anonymous="allow", key=args.wandb_key)
    sdec = SDECPipeline()
    sdec.test_model(args.trained_checkpoint,
                    args.num_labels, 
                    args.test_type,
                    test_noise=args.label_noise,
                    dataset_type=args.dataset_type, 
                    santa_barbara_path=args.santa_barbara_path, 
                    write_csv=args.write_csv, 
                    csv_save_path=args.csv_save_path)

if __name__ == "__main__":
    main()