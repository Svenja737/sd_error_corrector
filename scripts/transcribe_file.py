from sdec_pipeline import SDECPipeline

def main():

    sdec = SDECPipeline()
    sdec.transcribe_audio_file("/home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/raw/gkiki.wav", "cKlZoSBS9eATlIW1VhK_QWMXz0aH3ej9_iUGt7x1Xefl")

if __name__ == "__main__":
    main()