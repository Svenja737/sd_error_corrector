import csv
import os
from transformers import AutoTokenizer

class CSVWriter:
    """
    Class for saving information from training/eval/testing and writing them into a CSV file for inspection.

    Methods
    -------
    update_state(output_dict)
    clear_state()
    write()
    convert_ids_to_tokens()

    Attributes
    ----------
    state_list : list
    output_path : str
    """

    def __init__(self) -> None:
        self.state_list = []
        self.output_path = "results"

    def update_state(self, output_dict):
        self.state_list.append(output_dict)

    def clear_state(self):
        self.state_list = []

    def write_csv(self):
        field_names = self.state_list[0].keys()
        print(field_names)
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        with open(f"{self.output_path}/results.csv", "w", encoding="utf-8", newline="") as res_file:
            writer = csv.DictWriter(res_file, fieldnames=field_names, delimiter=";")
            writer.writeheader()
            writer.writerows(self.state_list)

    def convert_ids_to_tokens(self, input_ids_batch):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        input_tokens = tokenizer.batch_decode(input_ids_batch)
        return input_tokens
    
