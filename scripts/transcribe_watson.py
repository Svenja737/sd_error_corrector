from sd_error_correction import SDErrorCorrectionPipeline
import os

def main():

    sdcp = SDErrorCorrectionPipeline()
    #sdcp.transcribe_audio_file("ampme.wav", "/watson/ampme.json")
    t = sdcp.load_watson_results("/watson/ampme.json")
    print(t)

if __name__ == "__main__":
    main()