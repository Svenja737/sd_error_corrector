from sdec_pipeline import SDECPipeline

def main():

    sdec = SDECPipeline()
    sdec.transcribe_audio_file("/home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/raw/4065.wav", "")

if __name__ == "__main__":
    main()