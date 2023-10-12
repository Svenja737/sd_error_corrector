from sdec_pipeline import SDECPipeline
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("audio_file_to_transcribe", help="Path to an audio file (wav format).")
    parser.add_argument("watson_key", help="Authentification key from your IBM Profile.")
    parser.add_argument("--model_name", default="en-US_Multimedia", help="Watson STT Model to use for the transcription.")
    args = parser.parse_args()

    sdec = SDECPipeline()
    sdec.transcribe_audio_file(args.audio_file_to_transcribe, args.watson_key, args.model_name)

if __name__ == "__main__":
    main()