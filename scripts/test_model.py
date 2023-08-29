from sd_error_correction import SDErrorCorrectionPipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    sdcp = SDErrorCorrectionPipeline()
    sdcp.run_evaluation("roberta-base", num_labels=3)

if __name__ == "__main__":
    main()