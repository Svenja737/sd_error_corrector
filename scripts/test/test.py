from sdec_pipeline import SDECPipeline
import os
import wandb
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="roberta-base", help="Backbone model for extracting word embeddings (default: roberta-base).")
    parser.add_argument("num_labels", type=int, help="Number of labels, equal to max speakers in one sample in data.")
    parser.add_argument("trained_checkpoint", help="Trained SDEC model checkpoint.")
    parser.add_argument("dataset_type", choices=["switchboard", "santa_barbara", "fused"], help="Dataset variant, either Switchboard, Santa Barbara or both (i.e. fused).")
    parser.add_argument("wandb_key", help="Authentification key for wandb logger.")
    parser.add_argument("--testing_mode", default=None, help="Set to 'no_noise' for models trained in no_noise training mode.")
    parser.add_argument("--testing_noise", type=float, help=("Perturbation level for testing."))
    parser.add_argument("--santa_barbara_path", help="Path to downloaded SB dataset. Download at https://www.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7/files/sitefiles/research/SBC/SBCorpus.zip")
    parser.add_argument("--write_csv", action="store_true", help="Save model information during testing.")
    parser.add_argument("--csv_save_path", default="results", help="Location of the model information csv file.")
    args = parser.parse_args()

    wandb.login(anonymous="allow", key=args.wandb_key)
    sdec = SDECPipeline()
    sdec.test_model(args.model_name, 
                    args.num_labels, 
                    args.trained_checkpoint, 
                    testing_mode=args.testing_mode,
                    label_noise=args.testing_noise, 
                    dataset_type=args.dataset_type, 
                    santa_barbara_path=args.santa_barbara_path, 
                    write_csv=args.write_csv, 
                    csv_save_path=args.csv_save_path)

if __name__ == "__main__":
    main()