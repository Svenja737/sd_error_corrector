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
    parser.add_argument("watson_save_file", default="", help="File to Watson .json file.")
    parser.add_argument("--gold_label_file", help="File with tokens and reference labels (format: text file with 'token\tlabel\n' in each line.)")
    args = parser.parse_args()

    sdcp = SDECPipeline()

    text_file = args.watson_save_file.replace(".json", ".txt")

    data = sdcp.load_watson_results(args.watson_save_file)
    text_file_path = args.watson_save_file.replace(".json", ".txt")
    sdcp.save_watson_txt(data, text_file_path)

    gold_labels = sdcp.load_reference_from_txt(args.gold_label_file) 
    watson_labels = sdcp.load_reference_from_txt(text_file) 
    predictions = sdcp.inference(args.model_name, None, args.model_checkpoint, args.num_labels, data)
    print(predictions.size())


    watson_score = sdcp.score(watson_labels, gold_labels)
    print(watson_score)

    score_prediction = sdcp.score(predictions, gold_labels)
    print(score_prediction)
    

if __name__ == "__main__":
    main()