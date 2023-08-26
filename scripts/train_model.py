import pytorch_lightning as pl
from data_libs.sd_datamodule import SpeakerClassificationDataModule
from modeling.sd_classification import SpeakerDiarizationCorrectionModule
from pytorch_lightning.loggers import WandbLogger
import os
import wandb
from transformers import RobertaConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    sdm = SpeakerClassificationDataModule(model_name_or_path="roberta-base")
    sdm.setup("fit")
    sdm.setup("validate")

    model = SpeakerDiarizationCorrectionModule(
        model_name_or_path="roberta-base",
        num_labels=3
    )

    logger = WandbLogger(save_dir="results/wanddb_logging")

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        use_distributed_sampler=False,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=sdm.train_dataloader(), val_dataloaders=sdm.val_dataloader())

if __name__ == "__main__":
    main()