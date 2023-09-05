from typing import Dict
import json

"""Functions for processing output of IBM Watson STT Service of format:
{results_index,
results : list of dicts with ASR results,
speaker_labels: list of dicts of labels per word
}

goal: make output ready for SD correction.
"""

def read_watson_results(watson_results_file) -> Dict:

    tokens = []
    speaker_labels = []
    indiv_speakers = []

    with open(f"/home/sfilthaut/sd_error_corrector/sd_error_corrector/scripts/{watson_results_file}", "r") as file:
        watson_results = json.load(file)
    fields = watson_results.keys()
    if "speaker_labels" not in fields:
        print("WARNING: You are processing a file without speaker labels, please make sure you add them to your data.")
    
    for result in watson_results["results"]:
        for r in result["alternatives"]:
            for t in r["timestamps"]:
                tokens.append(t[0])

    for result in watson_results["speaker_labels"]:
        speaker_labels.append(int(result["speaker"])+1)
        indiv_speakers.append(int(result["speaker"])+1)
    
    
    num_speakers = len(set(indiv_speakers))

    return {
        "num_speakers" : num_speakers,
        "tokens" : tokens,
        "perturbed_labels" : speaker_labels
    }

def load_references(reference_text_file):

    labels = []
    with open(f"/home/sfilthaut/sd_error_corrector/sd_error_corrector/scripts/{reference_text_file}", "r") as file:
        for line in file.readlines():
            labels.append(int(line.split(" ")[1].strip("\n"))+1)

    return labels

