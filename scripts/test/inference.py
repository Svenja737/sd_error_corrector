from sdec_pipeline import SDECPipeline
import os
import torch 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    sdcp = SDECPipeline()
    #sdcp.transcribe_audio_file("/home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/q2ec7.wav")
    data = sdcp.load_watson_results("audio_samples/transcribed/q2ec7.json")
    # sdcp.save_watson_txt(data, "audio_samples/transcribed/4065.txt")
    gold_labels = sdcp.load_reference_from_txt("/audio_samples/corrected_sd/new_q2ec7.txt")
    corr_labels = [0 if label==3 else label for label in data["labels"]]
    watson_labels = torch.as_tensor([corr_labels])
    data["labels"] = corr_labels
    predictions = sdcp.inference("roberta-base", "models/epoch=20-step=25599.ckpt", data, 11)
    print(f"Watson labels: {watson_labels}")
    print(f"Model Predictions: {predictions}")
    score_prediction = sdcp.score(predictions, gold_labels)
    print(score_prediction)
    raw_score = sdcp.score(watson_labels, gold_labels)
    print(raw_score)
    

if __name__ == "__main__":
    main()