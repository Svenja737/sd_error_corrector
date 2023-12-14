# README

This is the repository of the 2023 Bachelor thesis project "Leveraging Large Language Models for
Speaker Diarization Error Correction". Below you can find instructions for replicating model training and testing.

## Installation

Install this module in editable mode by following these steps:

1. Clone this repository into an empty folder
2. Create a virtual environment for your experiments (make sure to use pip version 23.1. and python 3.11)\
Note: install the "data-pipelines" module first by following the instructions: https://github.com/mumair01/Data-Pipelines

3. Install the module with the command "python3 -m pip install -e sdec"

## Data Structure

SDEC (Speaker Diarization Error Corrector) currently supports two datasets for training and evaluation. You can select which you want to utilize in training or testing by changing the *dataset_name* parameter when executing training/testing in the CLI. The possible configurations are:

- "switchboard"
- "fused"

The "fused" dataset is a combination of the Switchbaord and Santa Barbara Corpus for American English. While the SwitchBoard corpus is automatically downloaded, the SantaBarbara corpus needs to be downloaded manually, as well as the path to your download adjusted in the training script [(Click here to get to their official website)](https://www.linguistics.ucsb.edu/research/santa-barbara-corpus). You will need only the annotations, not the audio files.

## Training

To replicate the model training done in this project, you can use the CLI or create a bash file (example available in *scripts/train*. Either way, before training you can adjust the parameters listed below

- model_name \
*Tested only using "roberta-base", but could probably run without issue with "roberta-large".*
- num_labels \
*The maximum number of speakers in a data sample. Set to 2 when using the Switchboard corpus, otherwise to 4 for the "fused" dataset.*
- dataset_type (options: ["switchboard", "fused"]) \
*The desired dataset configuration.*
- training_mode (options: "no_noise", "fixed_noise", "scheduled_noise", "overlap_noise") \
*The type of noise injected into the training data. For details see the "Noise" section below.*
- token_noise (options: "True", "False") \
*This type of noise combines the "overlap_noise" variant with additional token perturbations.*
- binary (options: "True", "False") \
*Set to "True" when training on the Switchboard dataset.*
- wandb_key \
*Logging for this project is set up using WandB. You need a profile to generate a key (if you do not have one, you can register [here](https://wandb.ai/site))*.
- label_noise \
*The amount of label noise to inject when using the "fixed_noise" option. Previous experiments were conducted with noise = [0.1, 0.2, 0.3].*
- santa_barbara_path \
*Path to downloaded Santa Barbara transcripts.*

To train an SDEC model you will need access to a GPU.

## Testing

To evaluate your own model or a pretrained model on the test sets, either use the command line directly or write a bash script containing the following parameters (examples at *scripts/test*):

- trained_checkpoint \
*A trained model checkpoint.*
- num_labels \
*Set to either 2 or 4 matching the number of labels used in model training.*
- dataset_type \
*Set to "switchboard" or "fused" to match training option.*
- test_type \
*See options for "training mode" above, same applies here.*
- test_noise \
*Same as "label_noise" in training options.*
- wandb_key \
*Authentification key for the WandB logger.*
- santa_barbara_path \
*Path to the santa barbara dataset (same as for training.)*
- write_csv \
*Use this option if you want a csv file with tokens, predicted labels and reference labels from a testing.*
- csv_save_path \
*If write_csv, define the location of the csv file.*

## Single Example Inference

In order to use a model on a single audio file, follow these steps (example at scripts/test):

1. Transcribe your audio file with using the transcription script provided in the "scripts" folder. Since SDEC currently supports IBM Watson only, you will need an IBM account (sign up [here](https://cloud.ibm.com/registration?target=%2Fdocs%2Fspeech-to-text%3Ftopic%3Dspeech-to-text-models-ng), if you do not have one), where you will be able to retrieve the required authentification token from your profile. Enter the token and all other required arguments for inference and run the script.
2. You will obtain a json file with the direct Watson output, as well as a text file with only the tokens and corresponding labels.

## Transcription

Inside the "scripts" folder, adjust and run the "transcribe.sh" file to get an IBM Watson transcription. An IBM account is required for this, in order to obtain an authentification token.

### IBM Watson

Obtaining a transcripiton from IMB Watson requires an account, as you will need an API access token. Once you have a key for Watson STT, enter it in he appropriate place when you run the
transcribe.sh script (for a direcory of files). You will receive *.json files wih he output that you can load with the provided functions in watson_utils.
