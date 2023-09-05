from sd_error_correction import SDErrorCorrectionPipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    sdcp = SDErrorCorrectionPipeline()
    data = sdcp.load_watson_results("/watson/single_examples/q2ec7.json")
    gold_labels = sdcp.load_reference_from_txt("/watson/single_examples/q2ec7_corrected.txt")
    sdcp.run_sd_error_correction("epoch=46-step=57293.ckpt", data, gold_labels)

if __name__ == "__main__":
    main()