import torch 
import random
import pytorch_lightning as L
import numpy as np
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
                 model_name_or_path: str,
                 num_labels: int, 
                 training_mode: str=None,
                 testing_mode: str=None,
                 token_noise: bool=False,
                 label_noise: float=None,
                 train_batch_size: int=8,
                 eval_batch_size: int=8,
                 learning_rate: float=1e-4,
                 adam_epsilon: float=1e-8,
                 warmup_steps: int=10,
                 weight_decay: float=0.0,
                 dropout_rate: float=0.1,
                 write_csv: bool=False,
                 csv_save_path: str=None
                 ) -> None:
        
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.training_mode = training_mode
        self.testing_mode = testing_mode
        self.token_noise = token_noise
        self.overlap_window = 3
        self.noise_near_n = 0.3
        self.noise_near_window = 4
        self.noise_far = 5
        self.num_labels = num_labels
        self.label_noise = label_noise
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout_rate = dropout_rate
        self.backbone = RobertaModel.from_pretrained(model_name_or_path)
        if self.training_mode == "no_noise":
            self.feature_dim = 768 
        else:
            self.feature_dim = 768 + self.num_labels
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.model = torch.nn.Linear(self.feature_dim, self.num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.metric = compute_metrics
        self.csv_save_path = csv_save_path
        self.csv_writer = CSVWriter(self.csv_save_path)
        self.write_csv = write_csv
        

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

        if self.token_noise == True:
            input_ids = self.perturb_tokens(input_ids)

        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)

        assert self.training_mode in ["fixed_noise", "scheduled_noise", "overlap_noise", "no_noise"], "Not a valid option for training!"

        if self.training_mode == "fixed_noise":
            assert self.label_noise != None, "Set a value for label_noise!"
            perturbed_labels = self.perturb_labels(p_labels, self.label_noise)
            fused_embeddings = self.reconcile_features_labels(backbone_embeddings, perturbed_labels)
        elif self.training_mode == "scheduled_noise":
            noise = self.schedule_random_noise_by_epoch()
            perturbed_labels = self.perturb_labels(p_labels, noise)
            fused_embeddings = self.reconcile_features_labels(backbone_embeddings, perturbed_labels)
        elif self.training_mode == "overlap_noise":
            perturbed_labels = self.perturb_labels_around_speaker_changes(p_labels, self.overlap_window)
            fused_embeddings = self.reconcile_features_labels(backbone_embeddings, perturbed_labels)
        elif self.training_mode == "no_noise":
            fused_embeddings = backbone_embeddings

        loss, logits = self(fused_embeddings, labels=labels)
        if self.training_mode == "fixed_noise":
            self.log("Label_Noise", self.label_noise, logger=True)
        if self.training_mode == "scheduled_noise":
            self.log("Label_Noise", noise, logger=True)
        if self.token_noise == True and self.current_epoch == 0: 
            self.log("Token Swap Example", self.csv_writer.convert_ids_to_tokens(input_ids), logger=True)
        self.log("Train_Loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions" : logits.argmax(dim=-1), "labels": labels}

    def validation_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        p_labels = batch["perturbed_labels"]
        labels = batch["labels"]
        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, p_labels)
        print(fused_embeddings)

        loss, logits = self(fused_embeddings, labels=labels)
        self.log("Val_Loss", loss, prog_bar=True, logger=True)
        self.validation_step_outputs.append({"val_loss": loss, "predictions" : logits.argmax(dim=-1), "labels": labels})
        return {"val_loss": loss, "predictions" : logits.argmax(dim=-1), "labels": labels}

    def test_step(self, batch, batch_ids):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        backbone_embeddings = self.get_embeddings(input_ids, attention_mask)
        fused_embeddings = self.reconcile_features_labels(backbone_embeddings, labels)
        logits = self(fused_embeddings)[1]

        self.test_step_outputs.append({"predictions" : logits.argmax(dim=-1), "labels": labels})
        if self.write_csv:
            tokens = self.csv_writer.convert_ids_to_tokens(input_ids)
            cleaned_preds, cleaned_labels = self.postprocess(logits.argmax(dim=-1), labels)
            self.csv_writer.update_state({"id" : batch_ids, 
                                          "tokens" : "".join(tokens), 
                                          "predictions" : str(cleaned_preds[0]),
                                          "labels" : str(cleaned_labels[0])})
            
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
        if self.write_csv:
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
    
    def perturb_labels(self, batched_labels, noise_n, test=False):
        """
        Parameters
        ----------
        labels : 
        noise_n : float

        Returns
        -------
        batch_perturbed : tensor
        """
        batch_perturbed = []
        for batch in torch.Tensor.tolist(batched_labels): 
            seq_length = len(batch)
            range_perturbed_labels = int(seq_length*noise_n)
            id_batch = [(i, label) for i, label in enumerate(batch)] 
            if test != True:
                random.shuffle(id_batch)
            label_list = [x for x in range(self.num_labels)]
            rand_labels = [(i[0], random.choice(label_list)) for i in id_batch[:range_perturbed_labels]]
            init_rand_labels = [0]*self.num_labels
            new_rand_labels = [(x[0], [y+1 if i==x[1] else y for i, y in enumerate(init_rand_labels)]) for x in rand_labels] 
            id_batch[:range_perturbed_labels] = new_rand_labels
            id_batch.sort()
            batch_perturbed.append([x[1] for x in id_batch])
        
        if torch.cuda.is_available():
            return torch.as_tensor(batch_perturbed, dtype=torch.int32, device="cuda")
        else:
            return torch.as_tensor(batch_perturbed, dtype=torch.int32, device="cpu")
        
    def perturb_labels_around_speaker_changes(self, batched_labels, win_size):
        """
        Keep label perturbations to sites of speaker overlap, where most labeling 
        errors tend to be located, as to not distort speaker patterns with random label allocations.
        """
        batch_perturbed = []
        for batch in torch.Tensor.tolist(batched_labels):
            perturbed = []
            mark_one = batch[0]
            mark_two = 0
            for i in range(len(batch)):
                if i < mark_two:
                    continue
                if mark_one == batch[i]:
                    perturbed.append(batch[i])
                    continue
                else:
                    perturbed += random.sample(batch[i:i+win_size], len(batch[i:i+win_size]))
                mark_one = batch[i]
                mark_two = i + win_size
            batch_perturbed.append(perturbed)

        if torch.cuda.is_available():
            return torch.as_tensor(batch_perturbed, dtype=torch.int32, device="cuda")
        else:
            return torch.as_tensor(batch_perturbed, dtype=torch.int32, device="cpu")

    def perturb_tokens(self, batched_input_ids):
        """
        Perturb tokens by swapping them a narrow context window and over longer 
        sentence distances. 
        noise_near_window: window size from which to sample
        noise_far: how many tokens to swap through the entire sequence
        """
        batch_perturbed = []
        for batch in torch.Tensor.tolist(batched_input_ids):
            perturbed = []
            for i in range(len(batch)):
                input_id_window = batch[i:i+self.noise_near_window]
                if -100 in input_id_window:
                    perturbed.append(input_id_window[0])
                    continue
                else:
                    change_token = random.choices([True, False], [self.noise_near_n, 1-self.noise_near_n], k=1)[0]
                    if change_token == True:
                        perturbed.append(random.sample(input_id_window, len(input_id_window))[0])
                    else:
                        perturbed.append(input_id_window[0])

            num_tokens = len(perturbed)
            index_list = random.sample(list(range(num_tokens)), self.noise_far)
            index_list_shuffled = random.sample(index_list, len(index_list))
            for j, k in list(zip(index_list, index_list_shuffled)):
                if j != -100 and k != -100:
                    perturbed[j], perturbed[k] = perturbed[k], perturbed[j]

        if torch.cuda.is_available():
            return torch.as_tensor(batch_perturbed, dtype=torch.int32, device="cuda")
        else:
            return torch.as_tensor(batch_perturbed, dtype=torch.int32, device="cpu")

    def schedule_random_noise_by_epoch(self):
        """
        Return increasing levels of noise, after an initial 3 epochs of training on 
        no noise. Increases noise every 3 epochs by 0.05.

        Parameters
        ----------

        Returns
        -------
        """
        start_noise = 0.0
        if self.current_epoch < 3:
            return start_noise
        elif self.current_epoch % 3 == 0:
            return float(self.current_epoch / 60)
    

