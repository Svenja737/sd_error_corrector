import torch 
import pytorch_lightning as L
from typing import Any, Dict, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from transformers import RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from performance_tracking.classification_metrics import compute_metrics
from performance_tracking.csv_writer import CSVWriter

class SDECModule(L.LightningModule):
    """
    Pytorch Lightning module for speaker diarization label re-classification.

    Attributes
    ----------
    model_name_or_path: str
        Path to a huggingface model in the hub or saved checkpoint (supported currently: roberta)
    num_labels: int
        Number of possible speakers in the training/evaluation data. 
    train_batch_size: int:
        Batch size for training (default: 8)
    eval_batch_size: int
        Batch size for evaluation (default: 8)
    learning_rate: float
        Learning rate for gradient descent (default: 1e-4)
    adam_epsilon: float
    warmup_steps: int
        Number of warm-up steps before training (default: 10)
    weight_decay: float
        Default: 0.0
    dropout_rate: float
        Default: 0.1

    Methods
    -------
    forward(fused_label_embeddings, labels=None)
        Forward step that makes a single set of predictions and returns loss/logits.
    get_embeddings(input_ids, attention_mask)
        Extract contextual embeddings from a pretrained language model.
    training_step(batch, batch_idx)
        Batched training step. 
    validation_step(batch, batch_idx)
        Batched validation step.
    test_step(batch, batch_idx)
        Batched test step.
    on_validation_epoch_end()
        Called at the end of a validation epoch to perform some operations.
    on_test_epoch_end()
        Called at the end of a test epoch to perform some operations.
    configure_optimizers()
        Set up optimization for training. 
    postprocess(predictions, labels)    
        Postprocess predictions and labels for scoring.
    reconcile_features_labels(backbone_embeddings, p_labels)
        Concatenate (perturbed) labels with word embeddings for fused features. 
    """

    def __init__(self, 
                 model_name_or_path,
                 num_labels=None, 
                 train_batch_size=8,
                 eval_batch_size=8,
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
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.model = torch.nn.Linear(self.feature_dim, self.num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.metric = compute_metrics
        self.csv_writer = CSVWriter()

    def forward(self, fused_labels_embeddings, labels=None):
        logits = self.model(fused_labels_embeddings)
        loss = None
        if labels != None:
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits
        
    def get_embeddings(self, input_ids, attention_mask):
        """
        Extract contextual word embeddings with the backbone model. 

        Parameters
        ----------
        input_ids
        attention_mask

        Returns
        -------
        sequence_outputs
        """
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        sequence_outputs = outputs[0]
        sequence_outputs = self.dropout(sequence_outputs)
        return sequence_outputs

    def training_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        loss, logits = self(fused_embeddings, labels=labels)
        print(f"Logits: {logits}")
        self.log("Train_Loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions" : logits.argmax(dim=-1), "labels": labels}

    def validation_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        loss, logits = self(fused_embeddings, labels=labels)
        self.log("Val_Loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({"val_loss": loss, "predictions" : logits.argmax(dim=-1), "labels": labels})
        return {"val_loss": loss, "predictions" : logits.argmax(dim=-1), "labels": labels}

    def test_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        logits = self(fused_embeddings)[1]
        self.test_step_outputs.append({"predictions" : logits.argmax(dim=-1), "labels": labels})
        tokens = self.csv_writer.convert_ids_to_tokens(input_ids)
        self.csv_writer.update_state({"id" : batch_ids, "tokens" : "".join(tokens), "logits" : str(torch.squeeze(logits, 0).tolist()), "predictions" : str(logits.argmax(dim=-1).tolist())})
        return {"predictions" : logits.argmax(dim=-1), "labels": labels}

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
        labels = []
        predictions = []
        for output in self.test_step_outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                predictions.append(out_predictions)
        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)
        true_labels, true_predictions = self.postprocess(predictions, labels)
        self.log_dict(self.metric(true_labels, true_predictions), logger=True)
        self.csv_writer.write_csv()
        self.csv_writer.clear_state()
        self.test_step_outputs.clear()
        return {"metrics" : self.metric(true_labels, true_predictions)}

    def configure_optimizers(self) -> Any:
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
        """
        Remove labels to be ignored in computing metrics (labels of special tokens).

        Parameters
        ----------
        predictions
        labels

        Returns
        -------
        true_labels
        true_predictions
        """
        true_labels = [[int(l.item()) for l in label if l != -100] for label in labels]
        true_predictions = [[int(p.item()) for (p, l) in zip(prediction, label) if l != -100]
                            for prediction, label in zip(predictions, labels)]

        return true_labels, true_predictions

    def reconcile_features_labels(self, backbone_embeddings, p_labels):
        """
        Combine backbone embeddings with predicted - potentially wrong - speaker labels.

        Parameters
        ----------
        backbone_embeddings : tensor
            Backbone word embeddings of size [BATCH_SIZE, MAX_SEQ_LENGTH, EMBEDDING SIZE].
        p_labels : tensor
            (Noisy) speaker labels of size [BATCH_SIZE, MAX_SEQ_LENGTH, NUM LABELS]
        Returns
        -------
        torch.cat((backbone_embeddings, p_labels), -1) : tensor
            Concatenated word embeddings and labels. 
        """
        return torch.cat((backbone_embeddings, p_labels), -1)
