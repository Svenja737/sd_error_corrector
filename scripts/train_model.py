import pytorch_lightning as pl
from data_libs.sd_data import SpeakerClassificationDataModule
from modeling.sd_model import SDCorrectionModel
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    sdm = SpeakerClassificationDataModule(model_name_or_path="roberta-base")
    sdm.setup("fit")
    sdm.setup("validate")

    model = SDCorrectionModel(
        model_name_or_path="roberta-base",
        num_labels=3
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        devices="auto",
        use_distributed_sampler=False,
        log_every_n_steps=3,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=sdm.train_dataloader(), val_dataloaders=sdm.val_dataloader())

if __name__ == "__main__":
    main()