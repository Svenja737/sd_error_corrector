import random
import data_pipelines.datasets as dp
import more_itertools as mit
import regex as re
from datasets import Dataset, DatasetDict
from data_pipelines import datasets
from typing import List, Dict
from data_pipelines.datasets import DataPipeline


# # # UTILS FOR SWICHBOARD CORPUS # # #

class SwitchboardPreprocessor:
    """
    Class containing methods for reading and preprocessing Switchboard annotated transcripts.

    Attributes
    ----------
    label_noise: float
        Level of perturbation of labels for feature combination during training.

    Methods
    -------
    add_speakers_to_turns(raw_switchboard_corpus)
        Adds speaker identities to separate speaking turns instead of just the session as a whole.
    sort_and_align(switchboard_corpus)
        Turns per session are originally saved separately for A, B. Combine turns for each session using timestamps.
    divide_sessions_into_chunks(switchboard_dataset, inference=False)
        Divide sessions into chunks of max. 512 tokens per chunk.
    max_item(dict)
        Determine number of chunks by dividing the longest session (i.e. most tokens) by 512 (max tokens)
    split_train_val_test(chunked_data)
        Splits data into train, val and test splits and returns it as a DatasetDict object.
    perturb_labels(label_list)
        For a given percentage of labels, randomly swap for another label.
    format_for_classification(switchboard_corpus)
        Execute preprocessing functions and filter out non-speech tokens.
    """

    def __init__(self, 
                 test_type: str=None,
                 test_noise: float=0.0,
                 test_set_overlap_window: int=3,
                 token_noise_win_size: int=4,
                 token_noise_probability: float= 0.8,
                 token_noise_far_swap: int=5,
                 ) -> None:
        
        self.test_type = test_type
        self.test_noise = test_noise
        self.test_set_overlap_window = test_set_overlap_window
        self.token_noise_win_size = token_noise_win_size
        self.token_noise_probability = token_noise_probability
        self.token_noise_far_swap = token_noise_far_swap

    def add_speakers_to_turns(self, raw_switchboard_corpus):
        """
        Add a speaker label to each turn in the corpus, instead each session.
        
        Parameters
        ----------

        Returns
        -------
        """

        updated_lines = []

        for line in raw_switchboard_corpus["full"]:
            updated_turns = []
            for turn in line["turns"]:
                turn["participant"] = line["participant"]
                updated_turns.append(turn)
            line["turns"] = updated_turns
            updated_lines.append(line)

        raw_switchboard_corpus["full"] = updated_lines
        return raw_switchboard_corpus
    
    def sort_and_align(self, switchboard_corpus) -> list:
        """
        Combine separate speakers of the same sessions into one instance.
        
        Parameters
        ----------

        Returns
        -------
        """

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
       
    def divide_sessions_into_chunks(self, switchboard_dataset, inference=False) -> list:
        """
        Divide Switchboard session into smaller chunks according to the default max length defined in the roberta configuration.
        
        Parameters
        ----------

        Returns
        -------
        """
        n_chunks = int(self.max_item(switchboard_dataset)/512) + 1
        if n_chunks == 0:
            n_chunks = 1

        chunked_data = []
        for i in switchboard_dataset:
            div_tokens = mit.divide(n_chunks, i["tokens"])
            div_labels = mit.divide(n_chunks, i["labels"])
            if inference==False:
                div_p_labels = mit.divide(n_chunks, i["perturbed_labels"])
                for tokens, labels, p_labels in list(zip(div_tokens, div_labels, div_p_labels)):
                    chunked_data.append({"tokens" : list(tokens), "labels" : list(labels), "perturbed_labels" : list(p_labels)})
            else:
                for tokens, labels in list(zip(div_tokens, div_labels)):
                    chunked_data.append({"tokens" : list(tokens), "labels" : list(labels)})

        for i, instance in enumerate(chunked_data): 
            instance.update({"id" : i+1})

        return chunked_data

    def max_item(self, dict) -> int:
        max_len = 0
        for i in dict:
            if max_len < len(i["tokens"]):
                max_len = len(i["tokens"])
        return max_len
    
    def split_train_val_test(self, chunked_data) -> DatasetDict:
        """
        Split Switchboard into a train, eval and test set (80/5/15)

        Parameters
        ----------

        Returns
        -------
        """

        train_len = int(0.8 * len(chunked_data))
        val_len = int(0.05 * len(chunked_data)) if int(0.05 * len(chunked_data)) != 0 else 1

        train_split = chunked_data[:train_len]
        random.shuffle(train_split)
        val_split = chunked_data[train_len:train_len+val_len]
        test_split = chunked_data[train_len+val_len:]

        if self.test_type == "fixed_noise":
            for item in test_split:
                item["perturbed_labels"] = self.perturb_test_labels(item["perturbed_labels"], self.test_noise)
        elif self.test_type == "overlap_noise":
            for item in test_split:
                item["perturbed_labels"] = self.perturb_test_labels_overlap(item["perturbed_labels"], self.test_set_overlap_window)
        elif self.test_type == "overlap_token_noise":
            for item in test_split:
                item["tokens"] = self.perturb_test_tokens(item["tokens"], self.token_noise_win_size, self.token_noise_probability, self.token_noise_far_swap)
        else:
            for item in test_split:
                item["tokens"] = item["tokens"]
                item["perturbed_labels"] = item["perturbed_labels"]

        prepared_data = DatasetDict({"train" : Dataset.from_list(train_split), 
                                    "validation" : Dataset.from_list(val_split), 
                                    "test" : Dataset.from_list(test_split)})

        return prepared_data
    
    def perturb_test_labels(self, perturbed_labels, noise_n):
        seq_length = len(perturbed_labels)
        range_perturbed_labels = int(seq_length*noise_n)
        id_batch = [(i, label) for i, label in enumerate(perturbed_labels)] 
        random.shuffle(id_batch)
        label_list = list(set(perturbed_labels))
        rand_labels = [(i[0], random.choice(label_list)) for i in id_batch[:range_perturbed_labels]]
        id_batch[:range_perturbed_labels] = rand_labels
        id_batch.sort()
        return [i[1] for i in id_batch]

    def perturb_test_labels_overlap(self, perturbed_labels, win_size):
        perturbed = []
        mark_one = perturbed_labels[0]
        mark_two = 0
        for i in range(len(perturbed_labels)):
            if i < mark_two:
                continue
            if mark_one == perturbed_labels[i]:
                perturbed.append(perturbed_labels[i])
                continue
            else:
                perturbed += random.sample(perturbed_labels[i:i+win_size], len(perturbed_labels[i:i+win_size]))
            mark_one = perturbed_labels[i]
            mark_two = i + win_size
        return perturbed

    def perturb_test_tokens(self, tokens, near_window_size, true_false_prob, noise_far):
        perturbed = []
        for i in range(len(tokens)):
            input_id_window = tokens[i:i+near_window_size]
            change_token = random.choices([True, False], [true_false_prob, 1-true_false_prob], k=1)[0]
            if change_token == True:
                perturbed.append(random.sample(input_id_window, len(input_id_window))[0])
            else:
                perturbed.append(input_id_window[0])

        num_tokens = len(tokens)
        index_list = random.sample(list(range(num_tokens)), noise_far)
        index_list_shuffled = random.sample(index_list, len(index_list))
        for j, k in list(zip(index_list, index_list_shuffled)):
            perturbed[j], perturbed[k] = perturbed[k], perturbed[j]

        return perturbed

    def format_for_classification(self, switchboard_corpus) -> DatasetDict:
        """
        Run functions for preparing switchboard data. Removes bracketed paralinguistic annotations.

        Parameters
        ----------

        Returns
        -------
        """

        m_temp = self.add_speakers_to_turns(switchboard_corpus)
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
                        speaker_labels.append(0)
                    elif turn["participant"] == "B":
                        speaker_labels.append(1)
                
            session_dict["session_id"] = turn["session_id"]
            session_dict["tokens"] = tokens
            session_dict["labels"] = speaker_labels
            session_dict["perturbed_labels"] = speaker_labels
            updated_data.append(session_dict)

        chunked = self.divide_sessions_into_chunks(updated_data)
        split_data = self.split_train_val_test(chunked)

        # split_data["train"].save_to_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sw_data/train")
        # split_data["validation"].save_to_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sw_data/validation")
        # split_data["test"].save_to_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sw_data/test")

        return split_data
    
