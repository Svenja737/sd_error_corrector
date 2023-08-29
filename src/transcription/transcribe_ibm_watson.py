import json
import os
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


def transcribe_audio(audio_path, output_path): 

    authenticator = IAMAuthenticator('cKlZoSBS9eATlIW1VhK_QWMXz0aH3ej9_iUGt7x1Xefl')
    service = SpeechToTextV1(
        authenticator=authenticator
    )

    service.set_service_url('https://api.eu-de.speech-to-text.watson.cloud.ibm.com/instances/9d805f04-9ca3-4bcb-bb21-662256068e8b')

    with open(audio_path, 'rb') as audio_file:
        # audio_source = AudioSource(audio_file)
        with open(f"/home/sfilthaut/sd_error_corrector/sd_error_corrector/scripts/{output_path}", "w") as file:
            json.dump(service.recognize(
                audio=audio_file,
                content_type='audio/wav',
                max_alternatives=0,
                speaker_labels=True,
                timestamps=True,
                profanity_filter=False,
                ).get_result(), file, indent=2)