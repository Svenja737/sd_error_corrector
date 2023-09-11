import random
import torch
import data_pipelines.datasets as dp
import itertools as it
import regex as re
from datasets import Dataset, DatasetDict
from typing import List, Dict


# # # UTILS FOR SWICHBOARD CORPUS # # #

class SwitchboardPreprocessor:

    def __init__(self, label_noise) -> None:
        self.label_noise = label_noise

    def add_speakers_to_turns(self, raw_switchboard_corpus):

        updated_lines = []

        for line in raw_switchboard_corpus["lines"]:
            updated_turns = []
            for turn in line["turns"]:
                turn["participant"] = line["participant"]
                updated_turns.append(turn)
            line["turns"] = updated_turns
            updated_lines.append(line)

        raw_switchboard_corpus["full"] = updated_lines
        return raw_switchboard_corpus
    
    def sort_and_align(self, switchboard_corpus):

        data = switchboard_corpus["full"]
        turns_by_session = []

        while(len(data)) > 1:
            turns_by_session.append([
                data[0]["session"], 
                data[0]["turns"], 
                data[1]["turns"]
                ])
            data = data[2:]

        all_turns = []
        for session_id, i, j in turns_by_session:
            session_turns = []
            for turns in i:
                session_turns.append({
            "session_id" : session_id,
            "participant" : turns["participant"],
            "start" : turns["start"],
            "end" : turns["end"],
            "text" : turns["text"]
        })
            for turns in j:
                session_turns.append({
            "session_id" : session_id,
            "participant" : turns["participant"],
            "start" : turns["start"],
            "end" : turns["end"],
            "text" : turns["text"]
        })
            
            sorted_session = sorted(session_turns, key=lambda x: (x["start"]))
            all_turns.append(sorted_session)

        return all_turns
    
    def divide_sessions_into_chunks(self, switchboard_dataset, inference=False):
        """swichboard sessions are long, so need to be divided into smaller sections so that they fit the 
        max_length of roberta (512). Though technically, I could reset that max length to max session length, 
        but hat would be very long.
        """
        n_chunks = int(self.max_item(switchboard_dataset)/512) if int(self.max_item(switchboard_dataset)/512) != 1 else int(self.max_item(switchboard_dataset)/512) + 1
        chunked_data = []
        for i in switchboard_dataset:
            div_tokens = it.divide(n_chunks, i["tokens"])
            div_labels = it.divide(n_chunks, i["labels"])
            if inference==False:
                div_p_labels = it.divide(n_chunks, i["perturbed_labels"])
                for tokens, labels, p_labels in list(zip(div_tokens, div_labels, div_p_labels)):
                    chunked_data.append({"tokens" : list(tokens), "labels" : list(labels), "perturbed_labels" : list(p_labels)})
            else:
                for tokens, labels in list(zip(div_tokens, div_labels)):
                    chunked_data.append({"tokens" : list(tokens), "labels" : list(labels)})

        for i, instance in enumerate(chunked_data): 
            instance.update({"id" : i+1})

        return chunked_data

    def max_item(self, dict):
        max_len = 0
        for i in dict:
            if max_len < len(i["tokens"]):
                max_len = len(i["tokens"])
        return max_len
    
    def split_train_val_test(self, chunked_data):

        train_len = int(0.8 * len(chunked_data))
        val_len = int(0.05 * len(chunked_data)) if int(0.05 * len(chunked_data)) != 0 else 1
        # print(train_len, val_len, test_len)

        train_split = chunked_data[ : train_len]
        random.shuffle(train_split)
        val_split = chunked_data[train_len : train_len+val_len]
        test_split = chunked_data[train_len+val_len : ]

        prepared_data = DatasetDict({"train" : Dataset.from_list(train_split), 
                                    "validation" : Dataset.from_list(val_split), 
                                    "test" : Dataset.from_list(test_split)})

        return prepared_data
    
    def perturb_labels(self, label_list):

        labels = list(set(label_list))
        id_labels = [(i, label) for i, label in enumerate(label_list)]

        random.shuffle(id_labels)
        num = int(self.label_noise*len(id_labels))
        rand_labels = [(i, random.choice(labels)) for i in id_labels[:num]]
        id_labels[:num] = rand_labels
        id_labels.sort()
        perturbed = [x[1] for x in id_labels]
        
        return perturbed

    def format_for_classification(self, switchboard_corpus):

        m_temp = self.modify_data(switchboard_corpus)
        s_temp = self.sort_and_align(m_temp)

        updated_data = []
        for session_list in s_temp:

            session_dict = {}
            speaker_labels = []
            tokens = []

            for turn in session_list:
                temp_tokens = []
                p = re.compile("\[[a-z]*\]")
                temp_tokens = [x for x in turn["text"].split(" ") if x != None and p.match(x) == None] # remove brackets [silence] and [noise]
                for t in temp_tokens:
                    tokens.append(t)
                    if turn["participant"] == "A":
                        speaker_labels.append(1)
                    elif turn["participant"] == "B":
                        speaker_labels.append(2)
                
            session_dict["session_id"] = turn["session_id"]
            session_dict["tokens"] = tokens
            session_dict["labels"] = speaker_labels
            session_dict["perturbed_labels"] = self.perturb_labels(speaker_labels)
            updated_data.append(session_dict)

        chunked = self.chunk_dataset(updated_data)
        split_data = self.split_train_val_test(chunked)

        return split_data
    
