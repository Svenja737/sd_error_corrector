from typing import Any, List, Optional, Tuple, Union
from pytorch_lightning import LightningModule
from transformers import AutoConfig, get_linear_schedule_with_warmup, AutoModelForTokenClassification
from data_libs.metrics import compute_metrics
import torch
from torch.optim import AdamW


class SDCorrectionModel(LightningModule):

    def __init__(
            self,
            model_name_or_path,
            num_labels: int = 3,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 10,
            weight_decay: float = 0.0,
            train_batch_size: int = 16,
            eval_batch_size: int = 16
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.label_names = ["NA", "A", "B"]
        self.label2id = {i: label for i, label in enumerate(self.label_names)}
        self.id2label= {v: k for k, v in self.label2id.items()}
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=self.num_labels, label2id=self.label2id, id2label=self.id2label)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = compute_metrics
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs[0]
        preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        self.validation_step_outputs.append({"loss": val_loss, "preds": preds, "labels": labels})
        self.log("val loss", val_loss, prog_bar=True, logger=True)
        return {"loss": val_loss, "preds": preds, "labels": labels}
    
    def on_validation_epoch_end(self):
        labels = []
        predictions = []
        for output in self.validation_step_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["preds"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        true_labels, true_predictions = self.postprocess(predictions, labels)
        print(self.metric(true_labels, true_predictions))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    

    def postprocess(self, predictions, labels):

        list_labels = labels.squeeze().tolist()
        list_preds = predictions.squeeze().tolist()

        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [[self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                            for prediction, label in zip(predictions, labels)]

        return true_labels, true_predictions