from sd_error_correction import SDErrorCorrectionPipeline
import os
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main():

    wandb.login(anonymous="allow", key="fa9a8ca94055c30f4a506593d5ed4d3a3cad70f2")
    sdcp = SDErrorCorrectionPipeline()
    sdcp.train_model("roberta-base", num_labels=3)

if __name__ == "__main__":
    main()