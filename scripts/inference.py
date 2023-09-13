from sdec_pipeline import SDECPipeline
import os
import torch 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    sdcp = SDECPipeline()
    #sdcp.transcribe_audio_file("/home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/q2ec7.wav")
    data = sdcp.load_watson_results("audio_samples/transcribed/q2ec7.json")
    gold_labels = sdcp.load_reference_from_txt("/audio_samples/corrected_sd/q2ec7_corrected.txt")
    watson_labels = torch.as_tensor([data["labels"]])
    print(watson_labels)
    predictions = sdcp.inference("roberta-base", "models/epoch=39-step=48760.ckpt", data, 3)
    score_prediction = sdcp.score(predictions, gold_labels)
    print(score_prediction)
    raw_score = sdcp.score(watson_labels, gold_labels)
    print(raw_score)
    

if __name__ == "__main__":
    main()