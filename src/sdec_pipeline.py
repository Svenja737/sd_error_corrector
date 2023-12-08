import torch 
import evaluate
import pytorch_lightning as L
from transformers import AutoTokenizer
from typing import Dict
from modelling.lightning_sd_model import SDECModule
from data_lib.lightning_data_module import SDDataModule
from pytorch_lightning.loggers import WandbLogger
from data_lib.data_prep_switchboard import SwitchboardPreprocessor
from transcription_tools.transcribe_audio import transcribe
from transcription_tools.utils_watson import read_watson, load_labels, save_as_txt
from performance_tracking.classification_metrics import compute_metrics

class SDECPipeline:
    """
    Pipeline for diarization label correction model training/evaluation, audio transcription and 
    diarization using IBM Watson and audio enhancement/noise reduction.

    Methods:
    --------
    train_model(model_name_or_path, num_labels, label_noise)
        train a classifier to correct speaker labels for a transcript.
    test_switchboard(model_name_or_path, checkpoint, num_labels, label_noise)
        test diarization re-classifier on specifically a switchboard test split.
    inference(model_name_or_path, checkpoint, inputs, num_labels)
        perform diarization label correction for single examples.
    score(preds, labels)
        return accuracy of diarization labels vs. reference labels
    """

    def __init__(self):
        pass

    # # # FUNCTIONS FOR MODEL TRAINING AND TESTING # # # 

    def train_model(self, 
                    model_name_or_path: str, 
                    num_labels: int, 
                    dataset_type: str, 
                    training_mode: str,
                    token_noise:bool,
                    label_noise: float=None,
                    santa_barbara_path: str=None) -> None:
        """
        Trains a speaker diarization label correction model.

        Parameters
        ----------
        model_name_or_path: str
            Valid name of a huggingface model or path to a saved model. 
        num_labels: int
            Number of speakers in the training data.
        label_noise: float
            Amount of perturbation on the labels fused with input tokens.
        """

        sdec_datamodule = SDDataModule(
            model_name_or_path,
            dataset_type,
            num_labels,
            santa_barbara_path=santa_barbara_path
            )
        
        sdec_model = SDECModule(
            model_name_or_path,
            training_mode=training_mode,
            token_noise=token_noise,
            num_labels=num_labels,
            label_noise=label_noise,
            train_batch_size=8,
            eval_batch_size=8
        )

        sdec_datamodule.setup("fit")
        sdec_datamodule.setup("validate")

        logger = WandbLogger(save_dir=f"wandb_logging/{model_name_or_path}")

        trainer = L.Trainer(
            devices=1,
            log_every_n_steps=10,
            max_epochs=100,
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=logger
        )

        trainer.fit(sdec_model, train_dataloaders=sdec_datamodule.train_dataloader(), val_dataloaders=sdec_datamodule.val_dataloader())

    def test_model(self, 
                   model_name_or_path, 
                   num_labels,
                   checkpoint, 
                   label_noise=None,
                   dataset_type=None, 
                   santa_barbara_path=None,
                   write_csv=False,
                   csv_save_path="results/results_test.csv"
                   ) -> None:
        """
        Evaluates a speaker diarization label correction model on the Switchboard dataset (test split).

        Parameters
        ----------
        model_name_or_path: str
            Name of a model used as the backbone (supported: roberta).
        checkpoint: str
            Path to a model checkpoint.
        inputs: dict
            Dictionary with tokens and labels.
        num_labels: int
            Number of possible labels in input data (default: 3).
        """

        sdec_datamodule = SDDataModule(
            model_name_or_path,
            dataset_type,
            num_labels,
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=4,
            santa_barbara_path=santa_barbara_path
            )
        
        sdec_model = SDECModule.load_from_checkpoint(
            checkpoint,
            label_noise=label_noise,
            map_location=torch.device('cpu'),
            num_labels=num_labels,
            train_batch_size=1,
            eval_batch_size=1,
            write_csv=write_csv, 
            csv_save_path=csv_save_path
        )

        sdec_datamodule.setup("test")

        logger = WandbLogger(save_dir=f"wandb_logging/{model_name_or_path}")

        trainer = L.Trainer(
            devices=1,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=logger,
        )

        trainer.test(sdec_model, dataloaders=sdec_datamodule.test_dataloader())

    def inference(self, 
                  model_name_or_path: str, 
                  dataset_type: str,
                  checkpoint: str,
                  num_labels: int, 
                  inputs: dict,
                  noise: str=False) -> list:
        """
        Performs inference on a single data instance.

        Parameters
        ----------
        model_name_or_path: str
            Name of a model used as the backbone (supported: roberta).
        checkpoint: str
            Path to a model checkpoint.
        inputs: dict
            Dictionary with tokens and labels.
        num_labels: int
            Number of possible labels in input data (default: 3).

        Returns
        -------
        preds: list
            List of label predictions for each input token.
        """
        sdec_datamodule = SDDataModule(
            model_name_or_path,
            dataset_type,
            num_labels,
            train_batch_size=8,
            eval_batch_size=8,
            num_workers=4
            )
        
        sdec_model = SDECModule.load_from_checkpoint(
            checkpoint,
            map_location=torch.device('cpu'),
            num_labels=num_labels,
            train_batch_size=8,
            eval_batch_size=8,
        )

        preprocessor = SwitchboardPreprocessor()
        inputs["perturbed_labels"] = inputs["labels"]
        chunked_data = preprocessor.divide_sessions_into_chunks([inputs])

        input_ids = []
        labels = []
        p_labels = []
        attention_mask = []
        preds = []

        for i in chunked_data:
            i["labels"] = [i["labels"]]
            i["perturbed_labels"] = [i["perturbed_labels"]]
            out = sdec_datamodule.tokenize_and_align_labels(i)
            input_ids.append(out["input_ids"])
            labels.append(torch.as_tensor(out["labels"]))
            p_labels.append(torch.as_tensor(out["perturbed_labels"]))
            attention_mask.append(out["attention_mask"])

        for i, a, l, p in list(zip(input_ids, attention_mask, labels, p_labels)):
            with torch.no_grad():
                embeddings = sdec_model.get_embeddings(i, a)
                fused = sdec_model.reconcile_features_labels(embeddings, p)
                outputs = sdec_model.forward(fused)
                preds.append(sdec_model.postprocess(outputs[1].argmax(dim=-1), l)[1])
            
        return torch.as_tensor(preds)
    
    def score(self, preds, labels) -> float:
        """
        Calculate accuracy of speaker labels predictions compared to reference labels. 

        Parameters
        ----------
        preds : tensor
            Tensor of predicted labels.
        labels : tensor
            Tensor of reference labels.

        Returns
        -------
        count/total : float
            Accuracy score derived from the fraction of correct predictions over all labels.
        """
        res = {}
        if preds.size()[0] > 1:
            all_preds_list = [p for p in preds]
            all_preds = torch.cat((all_preds_list[0], all_preds_list[1]), dim=1)
        else:
            all_preds = preds

        acc = evaluate.load("accuracy")
        prec = evaluate.load("precision")
        rec = evaluate.load("recall")
        f1 = evaluate.load("f1")

        res["accuracy"] = acc.compute(predictions=all_preds.squeeze().tolist(), references=labels.squeeze().tolist())["accuracy"]
        res["precision"] = prec.compute(predictions=all_preds.squeeze().tolist(), references=labels.squeeze().tolist(), average="weighted")["precision"]
        res["recall"] = rec.compute(predictions=all_preds.squeeze().tolist(), references=labels.squeeze().tolist(), average="weighted")["recall"]
        res["f1"] = f1.compute(predictions=all_preds.squeeze().tolist(), references=labels.squeeze().tolist(), average="weighted")["f1"]

        return res

    # # # FUNCTIONS FOR TRANSCRIPTION # # # 

    def transcribe_audio_file(self, audio_file, auth_token, model_name) -> None:
        """
        Transcribe an audio file with IBM Watson speech-to-text.
        """
        transcribe(audio_file, auth_token, model_name)
    
    def load_watson_results(self, output_path) -> Dict:
        """
        Load Watson results from a json file.
        """
        return read_watson(output_path)
    
    def load_reference_from_txt(self, path_to_corrected_labels) -> list:
        """
        Load corrected labels from a text file.
        """
        return load_labels(path_to_corrected_labels)
    
    def save_watson_txt(self, watson_output, filepath) -> None:
        """
        Save results from Watson (dict) in a text file.
        """
        save_as_txt(watson_output, filepath)
        
        


