echo "Transcription is in progress!"
python3 transcribe_file.py PATH/TO/WAV/FILE.wav \
                           "IBM WATSON KEY" \
                           --model_name "en-US_Multimedia" \
                           --watson_results_path LOCATION/FOR/SAVING/TRANSCRIPTION.json \
                           --text_file_path LOCATION/FOR/SAVING/TEXT/FILE/OF/RESULTS.txt \
                           > LOCATION/FOR/SAVING/TRANSCRIPTION.json
echo "Transcription complete!"