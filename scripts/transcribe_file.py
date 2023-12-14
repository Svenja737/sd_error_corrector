from sdec_pipeline import SDECPipeline
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("audio_file_to_transcribe", help="Path to an audio file (wav format).")
    parser.add_argument("watson_key", help="Authentification key from your IBM Profile.")
    parser.add_argument("--model_name", default="en-US_Multimedia", help="Watson STT Model to use for the transcription.")
    parser.add_argument("--watson_results_path", help="Location of the saved watson file.")
    parser.add_argument("--text_file_path", help="Save a text file of tokens and labels here.")
    args = parser.parse_args()

    sdec = SDECPipeline()
    sdec.transcribe_audio_file(args.audio_file_to_transcribe, args.watson_key, args.model_name)
    sdec.save_watson_txt(sdec.load_watson_results(args.watson_results_path), args.text_file_path)
    
if __name__ == "__main__":
    main()