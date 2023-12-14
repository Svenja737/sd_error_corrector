from sdec_pipeline import SDECPipeline
import os
import torch 
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="roberta-base", help="Name of backbone model.")
    parser.add_argument("model_checkpoint", help="Path to trained sdec model checkpoint.")
    parser.add_argument("num_labels", type=int)
    parser.add_argument("watson_text_file_path", help="File to Watson results text file.")
    parser.add_argument("--gold_label_file", help="File with tokens and reference labels (format: text file with 'token\tlabel\n' in each line.)")
    args = parser.parse_args()

    sdcp = SDECPipeline()
    watson_tokens, watson_labels = sdcp.load_watson_from_txt(args.watson_text_file_path)
    if args.gold_label_file != None: 
        correct_tokens, correct_labels = sdcp.load_watson_from_txt(args.gold_label_file)

    sdcp.inference(args.model_name, None, args.model_checkpoint, args.num_labels, watson_tokens, watson_labels, reference_labels=correct_labels, write_inference_csv=args.write_inference_csv, csv_path=args.csv_path)

if __name__ == "__main__":
    main()