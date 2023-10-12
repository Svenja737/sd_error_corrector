from typing import Dict
import json
import argparse
import torch


def read_watson(watson_results_file) -> Dict:
    """Reads a json file with IBM Watson results and returns it in the form of a dictionary.

    Parameters
    ----------
    watson_results_file: str
        Path to a json file with IBM Watson STT results. should have the fields "results" and "speaker_labels".

    Returns
    -------
    dict
        Dictionary with the number of individual speakers, tokens and labels in the file.
    """

    tokens = []
    speaker_labels = []
    indiv_speakers = []

    with open(watson_results_file, "r") as file:
        watson_results = json.load(file)
    fields = watson_results.keys()
    if "speaker_labels" not in fields:
        print("WARNING: You are processing a file without speaker labels, please make sure you add them to your data.")
    
    for result in watson_results["results"]:
        for r in result["alternatives"]:
            for t in r["timestamps"]:
                tokens.append(t[0])

    for result in watson_results["speaker_labels"]:
        speaker_labels.append(int(result["speaker"]))
        indiv_speakers.append(int(result["speaker"]))
    
    num_speakers = len(set(indiv_speakers))

    return {
        "num_speakers" : num_speakers,
        "tokens" : tokens,
        "labels" : speaker_labels
    }

def load_labels(reference_text_file) -> list:
    """
    Loads a list of labels from a textfile with corrected tokens and labels.

    Paramters
    ---------
    reference_text_file: list
        text file with tokens and speaker labels (separated by tab)

    Returns
    -------
    labels : list
        list of labels
    """

    labels = []
    with open(reference_text_file, "r") as file:
        for line in file.readlines():
            labels.append(int(line.split("\t")[1].strip("\n")))

    return torch.tensor([labels])


def load_tokens(reference_text_file) -> list:
    """
    Loads a list of tokens from a text file with corrected tokens and labels.

    Paramters
    ---------
    reference_text_file: list
        text file with tokens and speaker labels (separated by tab)

    Returns
    -------
    tokens : list
        list of tokens
    """

    tokens = []
    with open(f"/home/sfilthaut/sdec_revamped/sdec_revamped/{reference_text_file}", "r") as file:
        for line in file.readlines():
            tokens.append(line.split("\t")[0])

    return tokens


def save_as_txt(watson_output, filepath) -> None:
    """
    Saves results in a dictionary into a text file.

    Parameters
    ----------
    watson_output: dict
        dictionary with IBM Watson STT output, as returned by read_watson()
    filepath: str
        path to text file
    """
    with open(filepath, "w", encoding="utf-8") as file:
        for token, label in list(zip(watson_output["tokens"], watson_output["labels"])):
            file.write(f"{token}\t{label}\n")


