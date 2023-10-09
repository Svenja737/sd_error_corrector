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
    parser.add_argument("transcription_file")
    args = parser.parse_args()

    sdcp = SDECPipeline()
    # sdcp.transcribe_audio_file("/home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/sb_18.wav")
    data = sdcp.load_watson_results("audio_samples/transcribed/sb_18.json")
    # sdcp.save_watson_txt(data, "audio_samples/transcribed/sb_18_noisy.txt")
    #gold_labels = sdcp.load_reference_from_txt("/audio_samples/corrected_sd/sb_18_corrected.txt")
    watson_labels = sdcp.load_reference_from_txt("audio_samples/transcribed/sb_18.txt")
    predictions = sdcp.inference(args.model_name, "switchboard", args.model_checkpoint, args.num_labels, data)
    print(f"Watson labels: {watson_labels}")
    print(f"Model Predictions: {predictions}")
    # score_prediction = sdcp.score(predictions, gold_labels)
    # print(score_prediction)
    # raw_score = sdcp.score(watson_labels, gold_labels)
    # print(raw_score)
    

if __name__ == "__main__":
    main()