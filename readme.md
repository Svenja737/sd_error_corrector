# README

## Folder Structure

- pyproject
- config
- requirements
- readme
- scripts
  - train
  - eval
  - inference
  - transcribe
- src
  - data_libs
    - data_module
    - data_prep
    - utils
  - modeling
    - sdec_lightning_module
    - transcription
      - transcribe_with_watson
      - watson_utils
  - audio_enhancement
    - enhance_audio_file
    - audio_utils
  - SDEC_Pipeline

## Installation

Install this module in editable mode by following these steps:

1. Clone this repository into an empty folder
2. Create a virtual environment for your experiments (make sure to use pip version 23.1. and python 3.11)\
Note: install the "data-pipelines" module first by following the instructions: https://github.com/mumair01/Data-Pipelines

3. Install the module with the command "python3 -m pip install -e sdec"

## Data Structure

SDEC (Speaker Diarization Error Corrector) currently supports two datasets for training and evaluation. You can select which you want to utilize in training or testing by changing the *dataset_name* parameter in the file train.py. Possible configurations:

- "switchboard"
- "santa-barbara"
- "fused"

The "fused" dataset is a combination of both corpora. While the SwitchBoard corpus is automatically downloaded, the SantaBarbara corpus needs to be downloaded manually, as well as the path to your download adjusted in the training script. This is the link to their official website: https://www.linguistics.ucsb.edu/research/santa-barbara-corpus. You will need only the annotations, not the audio files.

## Training (WIP, to be detailed soon!)

To replicate the model training done in this project, simply you can run the bash script for training or the python script directly. Either way, before training you need to adjust the parameters in train.py, which are listed below

- model_name_or_path : str (options: ["roberta-base", "roberta-large"])
- num_labels : int
- max_epochs: int
- dataset_name : str (options: ["switchboard", "santa_barbara", "fused"])

optional::

- label_noise: float
- santa_barbara_path : str

To train an SDEC model you will need access to a GPU.

## Testing

To evaluate your own model or a pretrained model on the test sets, either use the command line directly or write a bash script containing the following parameters:

- model_name_or_path: should ordinarily be "roberta-base"
- num_labels: should match num_labels used during training
- trained_checkpoint: your trained model
- testing_mode: set to "no_fused_embeddings" when testing models that were trained using only the backbone embeddings
- dataset_type: same as in training, should match the training set
- wandb_key: authentification key for the wandb logger
- santa_barbara_path: path to the santa barbara dataset (same path as in training, as split is still done internally)
- write_csv: include, if you want a csv file with tokens, predicted labels and reference labels from testing
- csv_save_path: if write_csv, define the location of the csv file

## Single Example Inference

In order to use a model on a single audio file, follow these steps:

1. Transcribe your audio file with using the transcription script provided in the "scripts" folder. Since SDEC currently supports IBM Watson only, you will need an IBM account, where you will be able to retrieve the required authentification token from your profile. Enter the token and all other required arguments for inference and run the script.
2. You will obtain a json file with the direct Watson output, as well as a text file with only the tokens and corresponding labels.

## Transcription

Inside the "scripts" folder, adjust and run the "transcribe.sh" file to get an IBM Watson transcription. An IBM account is required for this, in order to obtain an authentification token.

### IBM Watson

Obtaining a transcripiton from IMB Watson requires an account, as you will need an API access token. Once you have a key for Watson STT, enter it in he appropriae place when you run the.
ranscrbe.sh script (for a direcory of files). You will receive json files wih he ouput that you can load with the provided funcions in Watson Utils.

Include a simple speaker profiles into label prediction, with embeddings from all previously spoken words aggregated, to calculate possibility of speaker having uttered that word
