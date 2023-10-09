import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


def transcribe(audio_path: str, auth_token: str): 
    """Transcribe an audio file using IBM Watson websockets.

    Parameters:
    -----------
    audio_path: str
        filepath to audio file you want to transcribe
    auth_token: str
        IBM authenticator token for API
    """

    authenticator = IAMAuthenticator(auth_token)
    service = SpeechToTextV1(
        authenticator=authenticator
    )

    service.set_service_url("https://api.eu-de.speech-to-text.watson.cloud.ibm.com/instances/9d805f04-9ca3-4bcb-bb21-662256068e8b")

    class MyRecognizeCallback(RecognizeCallback):

        def __init__(self):
            RecognizeCallback.__init__(self)

        def on_data(self, data):
            print(json.dumps(data, indent=2))

        def on_error(self, error):
            print('Error received: {}'.format(error))

        def on_inactivity_timeout(self, error):
            print('Inactivity timeout: {}'.format(error))

    my_recognize_callback = MyRecognizeCallback()

    with open(audio_path, 'rb') as audio_file:
        audio_source = AudioSource(audio_file)
        service.recognize_using_websocket(
            audio=audio_source,
            recognize_callback=my_recognize_callback,
            model="en-US_Multimedia",
            content_type='audio/wav',
            max_alternatives=1,
            speaker_labels=True,
            timestamps=True,
            profanity_filter=False,)