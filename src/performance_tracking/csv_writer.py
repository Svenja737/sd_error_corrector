import csv
import os
import pandas as pd
from transformers import AutoTokenizer

class CSVWriter:
    """
    Class for saving information during testing and writing them into a CSV file for inspection.
    """

    def __init__(self, csv_save_path) -> None:
        self.state_list = []
        self.output_path = csv_save_path

    def update_state(self, output_dict):
        self.state_list.append(output_dict)

    def clear_state(self):
        self.state_list = []

    def write_csv(self):
        field_names = self.state_list[0].keys()

        df = pd.DataFrame.from_records(self.state_list)
        df.to_csv(self.output_path, columns=field_names, sep=";")

    def convert_ids_to_tokens(self, input_ids_batch):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        input_tokens = tokenizer.batch_decode(input_ids_batch)
        return input_tokens
    

    
