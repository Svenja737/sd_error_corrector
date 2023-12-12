import pandas as pd
import numpy as np
import matplotlib as mpl
from datasets import load_from_disk

"""
- number of datapoints
- max number of tokens 
- min number of tokens
- average tokens 
- balance of speakers per label (as in, is there often a predominant speaker)
- semantic themes (word meanings)
"""

def get_random_sample_sentences(token_df, num_sentences):
    for i, token in enumerate(token_df[:num_sentences]):
        print(f"{i}: {token}")

def get_token_max(token_df):
    max_len = 0
    for i in token_df:
        if len(i) > max_len:
            max_len = len(i)

    return max_len

def get_token_min(token_df):
    min_len = len(token_df[0])
    for i in token_df:
        if len(i) == 0:
            print(i)
        if len(i) < min_len:
            min_len = len(i)

    return min_len

def get_token_average(token_df):
    len_sum = 0
    for i in token_df:
        len_sum += len(i)
    return int(np.round(len_sum/len(token_df), 0))

def check_duplicates(df1, df2):
    for i, j in list(zip(df1, df2)):
        i.sort()
        j.sort()
        if i == j:
            print("Duplicate detected!")
            print(i)
            print(j)

def count_label_distribution(label_df):
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    label_dist = []
    for i in labels:
        is_i = 0
        for l in label_df:
            if len(list(set(l))) == i:
                is_i += 1

        label_dist.append((i, is_i))

    return label_dist

sw_train = pd.DataFrame(load_from_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sw_data/train"))
sw_val = pd.DataFrame(load_from_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sw_data/validation"))
sw_test = pd.DataFrame(load_from_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sw_data/test"))
sb_train = pd.DataFrame(load_from_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sb_data/train"))
sb_val = pd.DataFrame(load_from_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sb_data/validation"))
sb_test = pd.DataFrame(load_from_disk("/home/sfilthaut/sdec_revamped/sdec_revamped/sb_data/test"))

# print(len(sw_train), len(sw_val), len(sw_test))
# print(len(sb_train), len(sb_val), len(sb_test))

for i in sb_train["labels"]:
    print(list(set(i)))

# sw_train.to_csv("/home/sfilthaut/sdec_revamped/sdec_revamped/src/data_lib/sw_train.csv", sep=";")
# sw_val.to_csv("/home/sfilthaut/sdec_revamped/sdec_revamped/src/data_lib/sw_val.csv", sep=";")
# sw_test.to_csv("/home/sfilthaut/sdec_revamped/sdec_revamped/src/data_lib/sw_test.csv", sep=";")
# sb_train.to_csv("/home/sfilthaut/sdec_revamped/sdec_revamped/src/data_lib/sb_train.csv", sep=";")
# sb_val.to_csv("/home/sfilthaut/sdec_revamped/sdec_revamped/src/data_lib/sb_val.csv", sep=";")
# sb_test.to_csv("/home/sfilthaut/sdec_revamped/sdec_revamped/src/data_lib/sb_test.csv", sep=";")

# print("Switchboard: ")
# print(get_token_max(sw_train["tokens"]))
# print(get_token_min(sw_train["tokens"]))
# print(get_token_average(sw_train["tokens"]))

# print(get_token_max(sw_val["tokens"]))
# print(get_token_min(sw_val["tokens"]))
# print(get_token_average(sw_val["tokens"]))

# print(get_token_max(sw_test["tokens"]))
# print(get_token_min(sw_test["tokens"]))
# print(get_token_average(sw_test["tokens"]))

# print("Santa Barbara: ")
# print(get_token_max(sb_train["tokens"]))
# print(get_token_min(sb_train["tokens"]))
# print(get_token_average(sb_train["tokens"]))

# print(get_token_max(sb_val["tokens"]))
# print(get_token_min(sb_val["tokens"]))
# print(get_token_average(sb_val["tokens"]))

# print(get_token_max(sb_test["tokens"]))
# print(get_token_min(sb_test["tokens"]))
# print(get_token_average(sb_test["tokens"]))

# get_random_sample_sentences(sw_test["tokens"], 5)
# get_random_sample_sentences(sb_test["tokens"], 5)

# check_duplicates(sw_train["tokens"], sw_test["tokens"])
# check_duplicates(sw_train["tokens"], sw_val["tokens"])
# check_duplicates(sw_val["tokens"], sw_test["tokens"])

# print(count_label_distribution(sb_train["labels"]))
# print(count_label_distribution(sb_val["labels"]))
# print(count_label_distribution(sb_test["labels"]))