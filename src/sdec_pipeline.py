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
from transcription_tools.utils_watson import read_watson, load_labels, save_as_txt, load_tokens
from performance_tracking.classification_metrics import compute_metrics
from performance_tracking.csv_writer import CSVWriter

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
                    binary:bool,
                    label_noise: float=None,
                    token_noise:bool=False,
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
            test_type=None,
            test_noise=0.0,
            santa_barbara_path=santa_barbara_path
            )
        
        print(f"Token Noise Pipeline: {token_noise}")
        
        sdec_model = SDECModule(
            model_name_or_path,
            training_mode=training_mode,
            num_labels=num_labels,
            label_noise=label_noise,
            token_noise=token_noise,
            binary=binary,
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
                   checkpoint, 
                   num_labels,
                   test_type,
                   test_noise=None,
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

        model_name = "roberta_base"

        sdec_datamodule = SDDataModule(
            model_name,
            dataset_type,
            num_labels,
            test_type,
            test_noise,
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=4,
            santa_barbara_path=santa_barbara_path
            )
        
        sdec_model = SDECModule.load_from_checkpoint(
            checkpoint,
            map_location=torch.device('cpu'),
            num_labels=num_labels,
            train_batch_size=1,
            eval_batch_size=1,
            write_csv=write_csv, 
            csv_save_path=csv_save_path
        )

        sdec_datamodule.setup("test")
        logger = WandbLogger(save_dir=f"wandb_logging/{model_name}")

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
                  watson_tokens: list,
                  watson_labels: list,
                  reference_labels=None,
                  write_inference_csv=False,
                  csv_path=None) -> list:
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
            test_type=None,
            test_noise=0.0,
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
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        inputs = {}
        inputs["tokens"] = watson_tokens
        inputs["perturbed_labels"] = watson_labels
        if reference_labels != None:
            inputs["labels"] = reference_labels
        else:
            inputs["labels"] = watson_labels

        chunked_data = preprocessor.divide_sessions_into_chunks([inputs])

        input_ids = []
        labels = []
        p_labels = []
        attention_mask = []
        preds = []
        p_label_list = []
        true_labels = []
        decoded_tokens = []

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
                p_label_list += self.post_process_perturbed_labels(p)
                true_labels += sdec_model.postprocess(outputs[1].argmax(dim=-1), l)[0][0]
                preds += sdec_model.postprocess(outputs[1].argmax(dim=-1), l)[1][0]
                decoded_tokens += tokenizer.batch_decode(i, skip_special_tokens=True)[0].split(" ")[1:]
                
        if write_inference_csv == True:
            csv = CSVWriter(csv_path)
            csv.update_state({
                "tokens" : "".join(decoded_tokens),
                "predictions" : str(preds[0]),
                "labels" : str(labels[0]),
                "perturbed_labels": p_label_list})
            csv.write_csv()

        if reference_labels != None:
            for t, l1, l2, l3 in list(zip(decoded_tokens, p_label_list, preds, true_labels)):
                print(f"Token: {t} | Watson Label: {l1} | Predicted Correction: {l2} | True Label: {l3}")  
            print(self.score(preds, true_labels, num_labels=num_labels))  
            print(self.score(torch.Tensor.tolist(p_labels), true_labels, num_labels=num_labels)) 
        else:
            for t, l1, l2 in list(zip(decoded_tokens, p_labels, preds)):
                print(f"Token: {t} | Watson Label: {l1} | Predicted Correction: {l2}")    


    def score(self, preds, labels, num_labels=2) -> float:
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

        acc = evaluate.load("accuracy")
        prec = evaluate.load("precision")
        rec = evaluate.load("recall")
        f1 = evaluate.load("f1")

        if num_labels == 2:
            res["accuracy"] = acc.compute(predictions=preds, references=labels.squeeze().tolist())["accuracy"]
            res["precision"] = prec.compute(predictions=preds, references=labels.squeeze().tolist(), average="binary")["precision"]
            res["recall"] = rec.compute(predictions=preds, references=labels.squeeze().tolist(), average="binary")["recall"]
            res["f1"] = f1.compute(predictions=preds, references=labels.squeeze().tolist(), average="binary")["f1"]
        else:
            res["accuracy"] = acc.compute(predictions=preds, references=labels.squeeze().tolist())["accuracy"]
            res["precision"] = prec.compute(predictions=preds, references=labels.squeeze().tolist(), average="weighted")["precision"]
            res["recall"] = rec.compute(predictions=preds, references=labels.squeeze().tolist(), average="weighted")["recall"]
            res["f1"] = f1.compute(predictions=preds, references=labels.squeeze().tolist(), average="weighted")["f1"]
       
        return res
    
    def post_process_perturbed_labels(self, p_labels):
        post_processed_labels = []
        for p in torch.Tensor.tolist(p_labels.squeeze()):
            if 1 in p:
                post_processed_labels.append(p.index(1))
        return post_processed_labels

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
    
    def load_watson_from_txt(self, path_to_corrected_file) -> (list, list):
        """
        Load tokens and labels from a text file.
        """
        return load_tokens(path_to_corrected_file), load_labels(path_to_corrected_file)
    
    def save_watson_txt(self, watson_output, filepath) -> None:
        """
        Save results from Watson (dict) in a text file.
        """
        save_as_txt(watson_output, filepath)
        
        


