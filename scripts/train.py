from sdec_pipeline import SDECPipeline
import os
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

def main():

    wandb.login(anonymous="allow", key="fa9a8ca94055c30f4a506593d5ed4d3a3cad70f2")
    sdec = SDECPipeline()
    sdec.train_model("roberta-base", 3, label_noise=0.0)

if __name__ == "__main__":
    main()