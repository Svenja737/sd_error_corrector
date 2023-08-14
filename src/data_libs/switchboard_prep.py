from data_pipelines.datasets import DataPipeline
from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoTokenizer
from random import shuffle
from tqdm import tqdm
import more_itertools as it
import regex as re
import os

class SwitchboardDataset:

    def __init__(self, dataset, variant):
        self.dataset = dataset
        self.variant = variant


    ### DATASET FORMATTING FOR TOKEN CLASSIFICATION ###
    def download_switchboard(self):

        dp = DataPipeline()
        switchboard_corpus = dp.load_dset(dataset=self.dataset, variant=self.variant)

        return switchboard_corpus


    def modify_switchboard(self, switchboard_corpus):

        updated_lines = []
        new_sw_corpus = switchboard_corpus
        
        for line in switchboard_corpus["full"]:
            updated_turns = []
            for turn in line["turns"]:
                turn["participant"] = line["participant"]
                for token in turn["tokens"]:
                    token["participant"] = line["participant"]
                updated_turns.append(turn)
            line["turns"] = updated_turns
            updated_lines.append(line)

        new_sw_corpus["full"] = updated_lines
        return new_sw_corpus
    

    def sort_and_align(self, new_sw_corpus):

        data = new_sw_corpus["full"]
        turns_by_session = []
        sorted_and_aligned = []

        while(len(data)) > 1:

            turns_by_session.append([
                data[0]["session"], 
                data[0]["turns"], 
                data[1]["turns"]
                ])

            data = data[2:]

        for session_id, i, j in turns_by_session:
            all_turns = []
            for turns in i:
                all_turns.append({
            "session_id" : session_id,
            "participant" : turns["participant"],
            "start" : turns["start"],
            "end" : turns["end"],
            "text" : turns["text"]
        })
        for turns in j:
            all_turns.append({
                "session_id" : session_id,
                "participant" : turns["participant"],
                "start" : turns["start"],
                "end" : turns["end"],
                "text" : turns["text"]
            })
    
        sorted_and_aligned.append(sorted(all_turns, key=lambda x: x["start"]))
        return sorted_and_aligned
        

    def split_into_chunks(self, dataset_split, num_chunks=5):
        
        chunked_data = []
        for i in dataset_split:
            div_tokens = it.divide(num_chunks, i["tokens"])
            div_labels = it.divide(num_chunks, i["spk_labels"])
            for tokens, labels in list(zip(div_tokens, div_labels)):
                chunked_data.append({"tokens" : list(tokens), "spk_labels" : list(labels), "session_id" : i["session_id"]})

        for i, instance in enumerate(chunked_data): 
            instance.update({"id" : i+1})

        return chunked_data
    

    def format_for_classification(self, sorted_and_aligned):

        updated_data = []
        for session_list in sorted_and_aligned:

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
                        speaker_labels.append(0)
                    elif turn["participant"] == "B":
                        speaker_labels.append(1)
                
            session_dict["session_id"] = turn["session_id"]
            session_dict["tokens"] = tokens
            session_dict["spk_labels"] = speaker_labels
            updated_data.append(session_dict)

        split_updated_data = self.split_into_chunks(updated_data)
        return split_updated_data


    def split_train_val_test(self, split_updated_data):

        train_len = 0.8 * len(split_updated_data)
        test_len = 0.15 * len(split_updated_data)

        train_split = split_updated_data[ : int(train_len)]
        shuffle(train_split)
        val_split = split_updated_data[int(train_len) : int(-test_len)]
        test_split = split_updated_data[int(-test_len) : ]

        prepared_data = DatasetDict({"train" : Dataset.from_list(train_split), 
                                     "val" : Dataset.from_list(val_split), 
                                     "test" : Dataset.from_list(test_split)})

        return prepared_data
    

    ### FUNCTIONS FOR DATA TOKENIZATION ###
    def align_labels_with_tokens(self, labels, word_ids):

        new_labels = []
        current_word = None

        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            elif word_id == current_word:
                new_labels.append(-100)

        return new_labels


    def tokenize_and_align_labels(self, examples):

        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
        all_labels = examples["spk_labels"]
        new_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels

        return tokenized_inputs
    

    def run_data_preparation(self):

        raw_dataset = self.download_switchboard()
        temp_1 = self.modify_switchboard(raw_dataset)
        temp_2 = self.sort_and_align(temp_1)
        formatted_dataset = self.format_for_classification(temp_2)
        split_formatted_data = self.split_train_val_test(formatted_dataset)

        tokenized_train = split_formatted_data["train"].map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=split_formatted_data["train"].column_names
        )

        tokenized_val = split_formatted_data["val"].map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=split_formatted_data["val"].column_names
        )

        tokenized_test = split_formatted_data["test"].map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=split_formatted_data["test"].column_names
        )
        return tokenized_train, tokenized_val, tokenized_test


        
    
