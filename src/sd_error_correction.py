"""Pipeline for running transcription, data preprocessing, training and model evaluation.
"""
from typing import Any, Dict, List
import pytorch_lightning as L
import torch
from transformers import AutoTokenizer
from transcription.transcribe_ibm_watson import transcribe_audio
from transcription.watson_utils import read_watson_results, load_references
from modeling.sd_classification import SpeakerDiarizationCorrectionModule 
from data_libs.sd_datamodule import SpeakerClassificationDataModule
from pytorch_lightning.loggers import WandbLogger
from data_libs.switchboard_utils import chunk_dataset
from data_libs.sd_datamodule import SpeakerClassificationDataModule
from data_libs.metrics import compute_metrics


class SDErrorCorrectionPipeline:

    def __init__(self):
        pass

    def transcribe_audio_file(self, audio_file, output_path) -> None:
        """
        Transcribe an audio file with IBM Watson speech-to-text.
        """
        transcribe_audio(audio_file, output_path)
    
    def load_watson_results(self, output_path) -> Dict:
        """
        Load Watson results from a json file.
        """
        return read_watson_results(output_path)
    
    def load_reference_from_txt(self, path_to_corrected_labels) -> List:
        """
        Load corrected labels from a text file.
        """
        return load_references(path_to_corrected_labels)
    
    def compare_sd_labels_predicted_corrected(self, predicted_labels: List, corrected_labels: List) -> int:
        # mostly for me so I have a reference point as to the expeced error rate of Watson SD labels
        count = 0
        total = len(predicted_labels)
        for p_label, r_label in list(zip(predicted_labels, corrected_labels)):
            if p_label == r_label:
                count += 1

        return count/total

    def preprocess_input_data(self):
        """Run data preprocessing for data, either Switchboard or Watson to send to train a model on or evaluate. 

        Return format: 
        {SPLIT: 
            [
                {id: id, tokens : [TOKENS], sd_labels : [SD_LABELS], labels : [LABELS]},
                ...
                {id: id, tokens : [TOKENS], sd_labels : [SD_LABELS], labels : [LABELS]}
            ]
        }
        """
        pass

    def train_model(self, model_name_or_path, num_labels):
        """
        Train an SD correction model for custom data, or finetune pretrained Switchboard classifier.
        
        Parameters:
        -----------
        model_name_or_path : str
            name of pretrained TokenClassification model or path to local custom model
        num_labels : int
            number of token classes 
        """
        sdc_datamodule = SpeakerClassificationDataModule(model_name_or_path)
        sdc_classifier = SpeakerDiarizationCorrectionModule(model_name_or_path, num_labels)

        sdc_datamodule.setup("fit")
        sdc_datamodule.setup("validate")

        logger = WandbLogger(save_dir="results/wanddb_logging")

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            use_distributed_sampler=False,
            log_every_n_steps=50,
            enable_progress_bar=True,
            logger=logger,
        )

        trainer.fit(sdc_classifier, train_dataloaders=sdc_datamodule.train_dataloader(), val_dataloaders=sdc_datamodule.val_dataloader())


    def run_evaluation(self, model_name_or_path):
        """´
        Perform evaluation for a previously trained SD Correction model, returning some metrics.

        Parameters:
        -----------
        model_name_or_path : str
            name of pretrained TokenClassification model or path to local custom model
        """
        sdc_datamodule = SpeakerClassificationDataModule("roberta-base")
        sdc_classifier = SpeakerDiarizationCorrectionModule.load_from_checkpoint(model_name_or_path, map_location=torch.device('cpu'))

        sdc_datamodule.setup("test")

        logger = WandbLogger(save_dir="results/wanddb_logging_test", offline=True)

        trainer = L.Trainer(
            accelerator="auto",
            use_distributed_sampler=False,
            log_every_n_steps=10,
            enable_progress_bar=True,
            logger=logger,
            check_val_every_n_epoch=1

        )

        trainer.test(sdc_classifier, dataloaders=sdc_datamodule.test_dataloader())

    def run_sd_error_correction(self, model_name_or_path, some_data, gold_labels):
        """
        Run error correction on a single transcribed example.

        Parameters:
        -----------
        model_name_or_path : str
            path to trained error correction model 
        data_example : dict
            dicionary with of format {tokens : [list of tokens], automatic_labels : [list of labels]}
        gold_labels : list
            list with (manually) corrected labels for the example

        Returns:
        --------
        preds : list of tensors
            predicions for corrected labels
        """
        sdc_datamodule = SpeakerClassificationDataModule("roberta-base")
        sdc_classifier = SpeakerDiarizationCorrectionModule.load_from_checkpoint(model_name_or_path, map_location=torch.device('cpu'))
        t = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, add_prefix_space=True)
        some_data["labels"] = gold_labels
        # labels = torch.tensor(some_data["labels"]) P_LABELS ARE ALREADY THE DISTURBED LABELS FROM TRANSCRIPTION!
        c_data = chunk_dataset([some_data])
        
        input_ids = []
        p_labels = []
        attention_mask = []
        labels = []
        preds = []
        for i in c_data:
            i["perturbed_labels"] = [i["perturbed_labels"]]
            i["labels"] = [i["labels"]]
            out = sdc_datamodule.tokenize_and_align_labels(i)
            input_ids.append(out["input_ids"])
            p_labels.append(out["perturbed_labels"])
            attention_mask.append(out["attention_mask"])
            labels.append(out["labels"])

        for i, a, p, l in list(zip(input_ids, attention_mask, p_labels, labels)):
            loss, output = sdc_classifier.forward(i, a, p, labels=l)
            preds.append(output.argmax(dim=-1))
            
        return preds


# p = read_watson_results("watson/single_examples/q2ec7.json")["perturbed_labels"]
# r = load_references("watson/single_examples/q2ec7_corrected.txt")
# C = SDErrorCorrectionPipeline()
# uevxo = C.load_watson_results("watson/single_examples/uevxo.json")
# for word, label in list(zip(uevxo["tokens"], uevxo["perturbed_labels"])):
#     print(word, label)
# out: 0.7636612021857924
