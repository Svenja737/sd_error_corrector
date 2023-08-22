import data_pipelines.datasets as dp
import more_itertools as it
import regex as re
from random import shuffle
from datasets import Dataset, DatasetDict
import random
from typing import List


def _modify_data(switchboard_corpus):
    updated_lines = []
    new_sw_corpus = switchboard_corpus
    
    for line in switchboard_corpus["full"]:
        updated_turns = []
        for turn in line["turns"]:
            turn["participant"] = line["participant"]
            updated_turns.append(turn)
        line["turns"] = updated_turns
        updated_lines.append(line)

    new_sw_corpus["full"] = updated_lines
    return new_sw_corpus

def _sort_and_align(new_sw_corpus):

    data = new_sw_corpus["full"]
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

def _chunk_dataset(dataset):

    n_chunks = int(_max_item(dataset)/512)

    chunked_data = []
    for i in dataset:
        div_tokens = it.divide(n_chunks, i["tokens"])
        div_labels = it.divide(n_chunks, i["labels"])
        div_p_labels = it.divide(n_chunks, i["perturbed_labels"])
        for tokens, labels, p_labels in list(zip(div_tokens, div_labels, div_p_labels)):
            chunked_data.append({"tokens" : list(tokens), "labels" : list(labels), "perturbed_labels" : list(p_labels), "session_id" : i["session_id"]})

    for i, instance in enumerate(chunked_data): 
        instance.update({"id" : i+1})

    return chunked_data

def _max_item(dict):
    max_len = 0
    for i in dict:
        if max_len < len(i["tokens"]):
            max_len = len(i["tokens"])
    return max_len

def _split_train_val_test(chunked_data):

    train_len = int(0.8 * len(chunked_data))
    val_len = int(0.05 * len(chunked_data)) if int(0.05 * len(chunked_data)) != 0 else 1
    # print(train_len, val_len, test_len)

    train_split = chunked_data[ : train_len]
    shuffle(train_split)
    val_split = chunked_data[train_len : train_len+val_len]
    test_split = chunked_data[train_len+val_len : ]

    prepared_data = DatasetDict({"train" : Dataset.from_list(train_split), 
                                    "validation" : Dataset.from_list(val_split), 
                                    "test" : Dataset.from_list(test_split)})

    return prepared_data

def format_for_classification(switchboard_corpus):
    # sorted_and_aligned = _sort_and_align(_modify_data(raw_switchboard_corpus))
    m_temp = _modify_data(switchboard_corpus)
    s_temp = _sort_and_align(m_temp)

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
        session_dict["perturbed_labels"] = perturb_labels(speaker_labels)
        updated_data.append(session_dict)

    chunked = _chunk_dataset(updated_data)
    split_data = _split_train_val_test(chunked)

    return split_data

def perturb_labels(label_list: List, noise_n: float=0.3) -> List:
    # simple label swap for noise_n split of instance labels 
    labels = list(set(label_list))
    id_labels = []
    rand_labels = []

    for i, label in enumerate(label_list):
        id_labels.append((i, label))

    random.shuffle(id_labels)
    num = int(noise_n*len(id_labels))
    for i, label in id_labels[:num]:
        label = random.choice(labels)
        id_labels.append((i, label))

    id_labels[:num] = rand_labels
    id_labels.sort()
    perturbed = [x[1] for x in id_labels]

    return perturbed





