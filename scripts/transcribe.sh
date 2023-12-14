echo "Transcription is in progress!"
python3 transcribe_file.py /home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/raw/mqtep.wav \
                           "cKlZoSBS9eATlIW1VhK_QWMXz0aH3ej9_iUGt7x1Xefl" \
                           --model_name "en-US_Multimedia" \
                           --watson_results_path /home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/transcribed/mqtep.json \
                           --text_file_path /home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/transcribed_as_textfile/mqtep.txt \
                           > /home/sfilthaut/sdec_revamped/sdec_revamped/audio_samples/transcribed/mqtep.json
echo "Transcription complete!"