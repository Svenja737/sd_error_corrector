from datasets import DatasetDict
from data_pipelines.datasets import DataPipeline
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from data_libs.switchboard_utils import format_for_classification
from data_libs.switchboard_utils import labels_to_vecs
import torch


class SpeakerClassificationDataModule(pl.LightningDataModule):

    TOKENIZERS_PARALLELISM = False

    loader_columns = [
        "input_ids",
        "perturbed_labels",
        "attention_mask",
        "token_type_ids",
        "labels"
    ]
    
    def __init__(
            self, 
            model_name_or_path: str,
            dataset: str = "switchboard",
            variant: str = "isip-aligned",
            train_batch_size: int = 8,
            eval_batch_size: int = 8,
            num_labels: int = 3,
            num_workers: int = 8,
            use_data_pipelines=True, 
            prepare_data_per_node=False,
            allow_zero_length_dataloader_with_multiple_devices=False
            ):
        
        super().__init__()
        self.use_dp = use_data_pipelines
        self.variant = variant
        self.dataset_name = dataset
        self.model_name_or_path = model_name_or_path
        self.prepare_data_per_node = prepare_data_per_node
        self.allow_zero_length_dataloader_with_multiple_devices = allow_zero_length_dataloader_with_multiple_devices
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_labels = num_labels
        self.dataset = None
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, add_prefix_space=True)

    def prepare_data(self) -> DatasetDict:
        if self.use_dp == True:
            dp = DataPipeline()
            dp.load_dset(dataset=self.dataset_name, variant=self.variant)
            AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, add_prefix_space=True)
            
    def setup(self, stage: str) -> DatasetDict:

        if self.use_dp == True:
            dp = DataPipeline()
            corpus = dp.load_dset(dataset=self.dataset_name, variant=self.variant)

        self.dataset = format_for_classification(corpus)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=self.dataset[split].column_names
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

            # modify perturbed labels to one-hot encoded vectors
            new_p = labels_to_vecs(self.dataset[split]["perturbed_labels"])
            self.dataset[split] = self.dataset[split].add_column(name="new_perturbed_labels", column=new_p)
            self.dataset[split] = self.dataset[split].remove_columns("perturbed_labels")
            self.dataset[split] = self.dataset[split].rename_column("new_perturbed_labels", "perturbed_labels")

        self.train_dataset = self.dataset["train"]
        # print(self.train_dataset)
        self.val_dataset = self.dataset["validation"]
        # print(self.val_dataset)
        self.test_dataset = self.dataset["test"]
        # print(self.test_dataset)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.data_collator)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)

    
    def align_labels_with_tokens(self, labels, word_ids, is_perturbed=False):
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
                if is_perturbed == True:
                    new_labels.append(labels[word_id])
                else:
                    new_labels.append(-100)
        return new_labels

    def tokenize_and_align_labels(self, examples, inference=False):
        # comment
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True, return_attention_mask=True)
        all_labels = examples["labels"]
        new_labels = []

        if inference==False:
            all_perturbed_labels = examples["perturbed_labels"]
            new_perturbed_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
            if inference==False:
                new_perturbed_labels.append(self.align_labels_with_tokens(all_perturbed_labels[i], word_ids, is_perturbed=True))

        tokenized_inputs["labels"] = new_labels
        if inference == False:
            tokenized_inputs["perturbed_labels"] = new_perturbed_labels
        return tokenized_inputs



