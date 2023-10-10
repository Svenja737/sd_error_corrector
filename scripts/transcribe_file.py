from sdec_pipeline import SDECPipeline
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("audio_file_to_transcribe", help="Path to an audio file (wav format).")
    parser.add_argument("watson_key", help="Authentification key from your IBM Profile.")
    args = parser.parse_args()

    sdec = SDECPipeline()
    sdec.transcribe_audio_file(args.audio_file_to_transcribe, args.watson_key)

if __name__ == "__main__":
    main()