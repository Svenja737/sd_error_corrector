import os
import random
import regex as re
from datasets import Dataset, DatasetDict
from data_lib.data_prep_switchboard import SwitchboardPreprocessor

class SantaBarbaraPreprocessor:

    def __init__(self) -> None:
        pass

    def read_corpus(self, file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as sb_file:
            lines = sb_file.readlines()
            turns = []
            for l in lines:
                split_line = l.split("\t")
                try:
                    if len(split_line) == 3:
                        turns.append({
                        "start" : split_line[0].split(" ")[0],
                        "speaker" : split_line[1].lower().strip(' ').strip(":"),
                        "tokens" : split_line[2].strip("\n")
                        })
                    else:
                        turns.append({
                        "start" : split_line[0],
                        "speaker" : split_line[2].lower().strip(' ').strip(":"),
                        "tokens" : split_line[3].strip("\n")
                        })
                except Exception:
                    continue

            return turns
        
    def extract_and_format(self, raw_data):
        all_speakers = []
        for turn_dict in raw_data:
            if type(turn_dict) == dict:
                speaker_id = turn_dict["speaker"]
                if speaker_id != '        ' and speaker_id not in all_speakers and speaker_id != ">env" and speaker_id != "":
                    all_speakers.append(speaker_id)
                turn_dict["tokens"] = self.remove_invalid_chars(turn_dict["tokens"].split(" "))
        return raw_data

    def remove_invalid_chars(self, list_of_words):
        clean_list = []
        invalid_chars = ["@", "-", "$", "[", "]", ".", "<", ">", "=", "--", "X", "&", "2", "3", "4", "~", "?", "%", "!"]
        invalid_expressions = ['', "VO", "YWN", "SING", "HI", "Q", "WHISTLE", "PAR", "TALK"]
        bracket_re = "[\(]+[A-Z]*_*[a-z]*[\)]+"
        for w in list_of_words:
            w_new = ""
            for char in w:
                if char not in invalid_chars:
                    w_new += char
            if w_new not in invalid_expressions and not re.match(bracket_re, w_new):
                clean_list.append(w_new)
    
        return clean_list
    
    def combine_split_speaker_turns(self, clean_data):
        turns_by_speaker = []
        current_speaker = None
        previous_speaker = None
        new_speaker_turns = {}
        for line in clean_data:
            if line["tokens"] != []:
                current_speaker = line["speaker"]
                if current_speaker == "" or current_speaker == ">env":
                    current_speaker = previous_speaker
                if current_speaker != previous_speaker:
                    new_speaker_turns = {}
                    new_speaker_turns.update({ 
                                            "start_time" : line["start"],
                                            "speaker" : current_speaker, 
                                            "tokens" : [line["tokens"]]
                                            })
                    
                    if new_speaker_turns != {}:
                        turns_by_speaker.append(new_speaker_turns)
                else:
                    try:
                        new_speaker_turns["tokens"].append(line["tokens"])
                    except KeyError:
                        new_speaker_turns["tokens"] = line["tokens"]

                previous_speaker = current_speaker

        for turn in turns_by_speaker:
            all_words = []
            for l in turn["tokens"]:
                for word in l:
                    all_words.append(word)
            turn["tokens"] = all_words

        return turns_by_speaker
        
    def normalize_labels(self, label_list):
        speakers = list(set(label_list))
        new_speakers = [speakers.index(l) for l in label_list]
        return new_speakers

    def make_dataset_object(self, path_to_data_files):
        switchboard = SwitchboardPreprocessor()
        corpus = []
        for file in os.listdir(path_to_data_files):
            if "Zone.Identifier" not in file:
                corpus.append(self.read_corpus(f"{path_to_data_files}/{file}"))

        cleaned_corpus = []
        for i, c in enumerate(corpus):
            cleaned_data = self.extract_and_format(c)
            combined_turns = self.combine_split_speaker_turns(cleaned_data)
            cleaned_corpus.append(combined_turns)
            
        corpus_by_file = []
        for i, file in enumerate(cleaned_corpus):
            all_file_data = {}
            all_file_tokens = []
            all_file_labels = []
            speaker_list = []

            for speaker_dict in file:
                all_file_tokens += speaker_dict["tokens"]
                speaker_list.append(speaker_dict["speaker"])
                speakers = list(set(speaker_list))
                curr_speaker = speakers.index(speaker_dict["speaker"])
                all_file_labels += [curr_speaker] * len(speaker_dict["tokens"])

            all_file_data = {
                "id" : i,
                "tokens" : all_file_tokens,
                "labels" : all_file_labels,
                "perturbed_labels" : all_file_labels
            }
            
            if all_file_data["tokens"] != []:
                corpus_by_file.append(all_file_data)
        
        chunked = switchboard.divide_sessions_into_chunks(corpus_by_file) 
        for c in chunked:
            c["labels"] = self.normalize_labels(c["labels"])
            c["perturbed_labels"] = self.normalize_labels(c["perturbed_labels"])
        dataset = switchboard.split_train_val_test(chunked)

        # dataset["train"].save_to_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sb_data/train")
        # dataset["validation"].save_to_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sb_data/validation")
        # dataset["test"].save_to_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sb_data/test")

        return dataset
    
# sb = SantaBarbaraPreprocessor()
# sb.make_dataset_object("/home/sfilthaut/sdec_revamped/SBCorpus/TRN")

            




