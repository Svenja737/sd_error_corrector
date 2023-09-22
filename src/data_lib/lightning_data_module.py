from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as L
from datasets import DatasetDict
from typing import Dict, List
from data_pipelines.datasets import DataPipeline
from torch.utils.data import DataLoader
from data_lib.data_prep import SwitchboardPreprocessor
from transformers import AutoTokenizer, DataCollatorForTokenClassification


class SDDataModule(L.LightningDataModule):
    """
    Pytorch Lightning data module for speaker diarization label re-classification data.
    Datasets are downloaded using M. Umair's pipeline: https://github.com/mumair01/Data-Pipelines

    Attributes
    ----------
    model_name_or_path: str
        Name of huggingface model in hub or saved locally (supported: roberta)
    train_batch_size: int
        Batch size for training (default: 8)
    eval_batch_size: int
        Batch size for evaluation (default: 8)
    num_labels: int
        Number of possible individual speakers in data.
    num_workers: int
    label_noise: float
        Level of reference label perturbation.
    prepare_data_per_node: bool
    allow_zero_length_dataloader_with_multiple_devices: bool
    dataset_name: str
        Name of dataset to download (default: "switchboard")
    variant: str 
        Variant of downloaded dataset (default: "isip-aligned")

    Methods
    -------
    setup(stage)

    train_dataloader()

    val_dataloader()

    test_dataloader()

    align_labels_with_tokens(labels, word_ids, is_perturbed=False)

    tokenize_and_align(examples)

    tokenize_and_align_labels_inference(examples)
        
    labels_to_vecs(batch_label_list)
    """

    TOKENIZERS_PARALLELISM = False

    loader_columns = [
        "input_ids", 
        "perturbed_labels",
        "attention_mask",
        "labels"
    ]

    def __init__(self,
                 model_name_or_path,
                 train_batch_size,
                 eval_batch_size,
                 num_labels,
                 num_workers,
                 label_noise,
                 prepare_data_per_node = False,
                 allow_zero_length_dataloader_with_multiple_devices=False,
                 dataset_name = "switchboard",
                 variant = "isip-aligned",
                 ) -> None:
        
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_labels = num_labels
        self.num_workers = num_workers
        self.label_noise = label_noise
        self.prepare_data_per_node = prepare_data_per_node
        self.allow_zero_length_dataloader_with_multiple_devices = allow_zero_length_dataloader_with_multiple_devices
        self.dataset_name = dataset_name
        self.variant = variant
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, add_prefix_space=True)


    def setup(self, stage: str) -> DatasetDict:
        dp = DataPipeline()
        corpus = dp.load_dset(dataset=self.dataset_name, variant=self.variant)

        switchboard_prep = SwitchboardPreprocessor(self.label_noise)
        self.dataset = switchboard_prep.format_for_classification(corpus)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=self.dataset[split].column_names
            )

            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["validation"]
        self.test_dataset = self.dataset["test"]
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.data_collator)
    

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
    

    def align_labels_with_tokens(self, labels, word_ids, is_perturbed=False) -> List:
        """
        Given token level labels and word_ids, align labels with tokens. 
        """
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                if is_perturbed==True:
                    label = 0 if word_id is None else labels[word_id]
                else:
                    label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                if is_perturbed==True:
                    new_labels.append(0)
                else:
                    new_labels.append(-100)
            elif word_id == current_word:
                if is_perturbed == True:
                    new_labels.append(labels[word_id])
                else:
                    new_labels.append(-100)
        return new_labels


    def tokenize_and_align_labels(self, examples) -> Dict:
        """
        Tokenize and align labels and perturbed labels with tokens.
        """
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True, return_attention_mask=True, return_tensors="pt")
        all_labels = examples["labels"]
        new_labels = []

        all_perturbed_labels = examples["perturbed_labels"]
        new_perturbed_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
            new_perturbed_labels.append(self.align_labels_with_tokens(all_perturbed_labels[i], word_ids, is_perturbed=True))
        tokenized_inputs["labels"] = new_labels
        tokenized_inputs["perturbed_labels"] = self.labels_to_vecs(new_perturbed_labels)
        return tokenized_inputs
    

    def tokenize_and_align_labels_inference(self, examples) -> Dict:
        """
        Tokenize and align labels (produced by STT engine) with tokens. 
        """
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True, return_attention_mask=True, return_tensors="pt")
        all_labels = examples["labels"]
        new_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = self.labels_to_vecs(new_labels)
        return tokenized_inputs
    

    def labels_to_vecs(self, batch_label_list) -> torch.tensor:
        """
        Turn a list of (batched) labels into a tensor of dimension (batchsize, seq_length, num_labels),
        where num_labels is the number of possible speakers in the dataset. 
        """
        label_vecs = []
        for label_list in batch_label_list:
            labels = len(list(set(label_list)))
            if labels != self.num_labels:
                labels = self.num_labels
            label_vec = [0] * len(label_list)
            label_vecs.append([[label_vec[i]+1 if i == j else label_vec[i] for j in range(labels)] for i in label_list])
        return torch.as_tensor(label_vecs)