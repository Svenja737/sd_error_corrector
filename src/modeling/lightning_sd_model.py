import torch 
import pytorch_lightning as L
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


class SDECModule(L.LightningModule):

    def __init__(self, 
                 model_name_or_path,
                 num_labels=None, 
                 train_batch_size=None,
                 eval_batch_size=None,
                 learning_rate=1e-4,
                 adam_epsilon=1e-8,
                 warmup_steps=10,
                 weight_decay=0.0,
                 dropout_rate=0.1,
                 ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout_rate = dropout_rate
        self.backbone = RobertaModel.from_pretrained(model_name_or_path)
        self.feature_dim = 768 + num_labels
        self.dropout = torch.nn.Dropout(self.dropout)
        self.model = torch.nn.Linear(self.feature_dim, self.num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        #self.metric = compute_metrics

    def forward(self, fused_labels_embeddings):
        logits = self.model(fused_labels_embeddings)
        return logits
        
    def get_embeddings(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_outputs = outputs[0]
        sequence_outputs = self.dropout(sequence_outputs)
        return sequence_outputs

    def training_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]

        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        print(p_labels.size())
        print(backbone_embeddings.size())
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        loss = self(fused_embeddings)[0]

        self.log("Train_Loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]

        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        print(p_labels.size())
        print(backbone_embeddings.size())
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        loss, outputs = self(fused_embeddings)

        self.log("Val_Loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({"val_loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels})
        return {"val_loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels}

    def test_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]

        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        print(p_labels.size())
        print(backbone_embeddings.size())
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        outputs = self(fused_embeddings)[1]

        self.test_step_outputs.append({"predictions" : outputs.argmax(dim=-1), "labels": labels})
        return {"predictions" : outputs.argmax(dim=-1), "labels": labels}

    def on_validation_epoch_end(self):
        labels = []
        predictions = []
        for output in self.validation_step_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        true_labels, true_predictions = self.postprocess(predictions, labels)
        self.log_dict(self.metric(true_labels, true_predictions), logger=True)
        self.validation_step_outputs.clear()
        return {"metrics" : self.metric(true_labels, true_predictions)}

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Any:
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
        true_labels = [[int(l.item()) for l in label if l != -100] for label in labels]
        true_predictions = [[int(p.item()) for (p, l) in zip(prediction, label) if l != -100]
                            for prediction, label in zip(predictions, labels)]

        return true_labels, true_predictions

    def reconcile_features_labels(self, backbone_embeddings, p_labels):
        pass

    def single_labels_to_vectors(self):
        pass