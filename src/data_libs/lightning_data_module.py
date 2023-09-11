from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import pytorch_lightning as L
from datasets import DatasetDict
from data_pipelines.datasets import DataPipeline
from torch.utils.data import DataLoader
from data_libs.data_prep import SwitchboardPreprocessor
from transformers import AutoTokenizer, DataCollatorForTokenClassification


class SDDataModule(L.LightningDataModule):

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
                 label_noise=0.3,
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
        corpus = dp.load_dset(dataset=self.dataset, variant=self.variant)

        switchboard_prep = SwitchboardPreprocessor(self.label_noise)
        self.dataset = switchboard_prep.format_for_classification(corpus)
        print(self.dataset)
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map()

            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["validation"]
        self.test_dataset = self.dataset["test"]
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.data_collator)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, collate_fn=self.data_collator)

    def align_labels_with_tokens(self, labels, word_ids):
        pass

    def tokenize_and_align_labels(self, examples):
        pass