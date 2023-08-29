import torch
import pytorch_lightning as L
from typing import Any, Optional
from typing import Dict
from pytorch_lightning.utilities.types import STEP_OUTPUT 
from transformers import RobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from data_libs.metrics import compute_metrics


class SpeakerDiarizationCorrectionModule(L.LightningModule):

    def __init__(self, 
                 model_name_or_path, 
                 num_labels: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 50,
                 weight_decay: float = 0.0,
                 train_batch_size: int = 16,
                 eval_batch_size: int = 16
                 ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.model = RobertaModel.from_pretrained(model_name_or_path)
        classifier_dropout = 0.1
        self.feature_dim = 769
        self.sdc = torch.nn.Linear(self.feature_dim, self.num_labels)
        print(f"SDC Size: {self.sdc}")
        self.dropout = torch.nn.Dropout(classifier_dropout)
        # TODO: implement BCELoss for multi-label classification
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.label_names = ["NA", "A", "B"]
        self.label2id = {i: label for i, label in enumerate(self.label_names)}
        self.id2label= {v: k for k, v in self.label2id.items()}
        self.metric = compute_metrics

    def forward(self, input_ids, attention_mask, p_labels, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        sequence_outputs = outputs[0]
        sequence_outputs = self.dropout(sequence_outputs)
        new_features = self.reconcile_features(sequence_outputs, p_labels)
        print(f"Last hidden layer size: {sequence_outputs.size()}")
        print(f"New features size: {new_features.size()}")
        logits = self.sdc(new_features)
        print(f"Logits: {logits.size()}")
        # logits = torch.sigmoid(outputs)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

    def training_step(self, batch, batch_idx) -> Dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, p_labels, labels)
        self.log("Train_Loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels}
    
    def validation_step(self, batch, batch_idx) -> Dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, p_labels, labels)
        self.log("Validation_Loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({"val_loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels})
        return {"val_loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels}
    
    def on_validation_epoch_end(self) -> None:
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
    
    def test_step(self, batch, batch_idx) -> Dict:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, p_labels, labels)
        self.log("Validation_Loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({"val_loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels})
        return {"test_loss": loss, "predictions" : outputs.argmax(dim=-1), "labels": labels}
    
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

        list_labels = labels.squeeze().tolist()
        list_preds = predictions.squeeze().tolist()

        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [[self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                            for prediction, label in zip(predictions, labels)]

        return true_labels, true_predictions
    
    def reconcile_features(self, roberta_embeddings_list, p_labels) -> torch.Tensor:
        return torch.cat((roberta_embeddings_list, torch.unsqueeze(p_labels, 2)), -1)

# hidden = torch.rand(1, 512, 768)
# p_labels = torch.unsqueeze(torch.rand(1, 512), 2)
# print(hidden.size())
# print(p_labels.size())
# print(torch.cat((hidden, p_labels), dim=-1))
# tok = RobertaTokenizerFast.from_pretrained("roberta-base", return_tensor="pt", return_attention_mask=True, add_prefix_space=True)
# model = RobertaModel.from_pretrained("roberta-base")

# inputs = [["Blue", "cats", "walk", "over", "the", "moon"], ["and", "they", "will", "fly", "away"]]
# labels = torch.tensor([[[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]])
# tok_ins = tok(inputs, is_split_into_words=True, padding=True)
# input_ids = torch.tensor(tok_ins["input_ids"])
# # print(input_ids)
# outputs = model(input_ids, output_attentions=True)
# h = outputs.last_hidden_state
# a = outputs.attentions
# t_ones = torch.ones((8, 3))

# start = time.time()
# new = torch.cat((h, labels), dim=-1)
# print(new[0].size())
# end = time.time()
# print(end - start)



