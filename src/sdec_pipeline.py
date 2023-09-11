import torch 
import pytorch_lightning as L
from transformers import AutoTokenizer
from modelling.lightning_sd_model import SDECModule
from data_lib.lightning_data_module import SDDataModule
from pytorch_lightning.loggers import WandbLogger

class SDECPipeline:

    def __init__(self):
        pass

    def train_model(self, model_name_or_path, num_labels):

        sdec_datamodule = SDDataModule(
            model_name_or_path,
            train_batch_size=8,
            eval_batch_size=8,
            num_labels=num_labels,
            num_workers=8,
            label_noise=0.3
            )
        
        sdec_model = SDECModule(
            model_name_or_path,
            num_labels=num_labels,
            train_batch_size=8,
            eval_batch_size=8
        )

        sdec_datamodule.setup("fit")
        sdec_datamodule.setup("validate")

        logger = WandbLogger(save_dir=f"wandb_logging/{model_name_or_path}")

        trainer = L.Trainer(
            devices=1,
            log_every_n_steps=30,
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=logger
        )

        trainer.fit(sdec_model, train_dataloaders=sdec_datamodule.train_dataloader(), val_dataloaders=sdec_datamodule.val_dataloader())
        


