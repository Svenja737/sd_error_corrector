"""Pipeline for running transcription, data preprocessing, training and model evaluation.
"""
from typing import Any
import pytorch_lightning as L
from transcription.transcribe_ibm_watson import transcribe_audio
from transcription.watson_utils import read_watson_results
from modeling.sd_classification import SpeakerDiarizationCorrectionModule 
from data_libs.sd_datamodule import SpeakerClassificationDataModule
from pytorch_lightning.loggers import WandbLogger


class SDErrorCorrectionPipeline:

    def __init__(self):
        pass

    def transcribe_audio_file(self, audio_file, output_path):
        """Permitted formats: see IBM Watson STT
        Output: IBM Watson STT transcription output. 
        """
        transcribe_audio(audio_file, output_path)
    
    def load_watson_results(self, output_path):
        return read_watson_results(output_path)

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
        """Train your own SD correction model for custom data, or finetune pretrained Switchboard classifier.
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
            log_every_n_steps=10,
            enable_progress_bar=True,
            logger=logger,
        )

        trainer.fit(sdc_classifier, train_dataloaders=sdc_datamodule.train_dataloader(), val_dataloaders=sdc_datamodule.val_dataloader())


    def run_evaluation(self, model_name_or_path):
        """´Perform evaluation for a previously trained SD Correction model, returning WDER metric.
        """
        sdc_datamodule = SpeakerClassificationDataModule(model_name_or_path)
        sdc_classifier = SpeakerDiarizationCorrectionModule(model_name_or_path, num_labels=3)

        sdc_datamodule.setup("fit")
        sdc_datamodule.setup("validate")

        logger = WandbLogger(save_dir="results/wanddb_logging")

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            use_distributed_sampler=False,
            log_every_n_steps=10,
            enable_progress_bar=True,
            logger=logger,
        )

        trainer.test(sdc_classifier, train_dataloaders=sdc_datamodule.train_dataloader(), val_dataloaders=sdc_datamodule.val_dataloader())

    def run_sd_error_correction(self, do_eval=False):
        """Run error correction without evaluation, using a pretrained model. 
        """
        pass

