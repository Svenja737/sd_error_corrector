"""Some utility functions to get statistics of the datasets

Desired stats:
- min/max/average length of audio (num frames / framerate (=frames per second))
- number of samples (= len(dataset["full"]))
- min/max/average number of speakers
- SNR?
- word-level timestamps yes/now?
- 
"""
from data_pipelines.datasets import DataPipeline
import os
import scipy
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment

def get_audio_length(dataset, path_to_audio_data_dir):
    lengths = []
    all_lengths = []
    for audio_file in os.listdir(path_to_audio_data_dir):
        file_ending = audio_file.split(".")[-1]
        if not file_ending == "wav":
            out_path = mp3_to_wav(audio_file, path_to_audio_data_dir, path_to_audio_data_dir+"_wav")
            sample_rate, data = wavfile.read(out_path)
        else:
            sample_rate, data = wavfile.read(path_to_audio_data_dir + "/" + audio_file)
        length = len(data)/sample_rate
        lengths.append({
            "filename" : audio_file,
            "length" : length
        })
        all_lengths.append(length)

    lengths.sort(key=lambda x: x["length"])
    min_length = lengths[0]
    max_length = lengths[-1]
    average_length = sum(all_lengths)/len(all_lengths)

    return {
        "Shortest audio" : min_length,
        "Longest audio" : max_length,
        "Average audio" : average_length,
        "All lengths" : lengths
    }

def mp3_to_wav(mp3_file_name, input_folder, output_folder):
    mp3_file_path = f"{input_folder}/{mp3_file_name}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    output_file_path = f"{output_folder}/{mp3_file_name[:-4]}.wav"
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(output_file_path, format="wav")
    return output_file_path

def get_dataset_columns(dataset):
    return dataset["full"].column_names

def get_dataset_size(dataset):
    return len(dataset["full"])

def get_num_speakers(dataset, dataset_designation: str="maptask"):
    pass

sample_pipeline = DataPipeline()
mt_data = sample_pipeline.load_dset(dataset="callfriend", variant="audio", language="eng-n")
#get_num_speakers(mt_data)
print(get_audio_length("callfriend", "/home/sfilthaut/.cache/data_pipelines/datasets/downloads/callfriend/callfriend_download/media/eng-n"))
