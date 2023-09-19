import torch 
import pytorch_lightning as L
from transformers import AutoTokenizer
from modelling.lightning_sd_model import SDECModule
from data_lib.lightning_data_module import SDDataModule
from pytorch_lightning.loggers import WandbLogger
from data_lib.data_prep import SwitchboardPreprocessor
from transcription_tools.transcribe_audio import transcribe
from transcription_tools.utils_watson import read_watson, load_labels, save_as_txt

class SDECPipeline:

    def __init__(self):
        pass

    # # # FUNCTIONS FOR MODEL TRAINING AND TESTING # # # 

    def train_model(self, model_name_or_path, num_labels, label_noise):

        sdec_datamodule = SDDataModule(
            model_name_or_path,
            train_batch_size=8,
            eval_batch_size=8,
            num_labels=num_labels,
            num_workers=4,
            label_noise=label_noise
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

    def test_switchboard(self, model_name_or_path, checkpoint, num_labels, label_noise):

        sdec_datamodule = SDDataModule(
            model_name_or_path,
            train_batch_size=8,
            eval_batch_size=8,
            num_labels=num_labels,
            num_workers=4,
            label_noise=label_noise
            )
        
        sdec_model = SDECModule.load_from_checkpoint(
            checkpoint,
            map_location=torch.device('cpu'),
            num_labels=num_labels,
            train_batch_size=8,
            eval_batch_size=8
        )

        sdec_datamodule.setup("test")

        logger = WandbLogger(save_dir=f"wandb_logging/{model_name_or_path}")

        trainer = L.Trainer(
            devices=1,
            log_every_n_steps=5,
            enable_progress_bar=True,
            enable_checkpointing=True,
            logger=logger
        )

        trainer.test(sdec_model, dataloaders=sdec_datamodule.test_dataloader())

    def inference(self, model_name_or_path, checkpoint, inputs, num_labels):

        sdec_datamodule = SDDataModule(
            model_name_or_path,
            train_batch_size=8,
            eval_batch_size=8,
            num_labels=num_labels,
            num_workers=4,
            label_noise=0.0
            )
        
        sdec_model = SDECModule.load_from_checkpoint(
            checkpoint,
            map_location=torch.device('cpu'),
            num_labels=num_labels,
            train_batch_size=8,
            eval_batch_size=8
        )

        preprocessor = SwitchboardPreprocessor(label_noise=0.0)
        chunked_data = preprocessor.divide_sessions_into_chunks([inputs], inference=True)

        input_ids = []
        p_labels = []
        attention_mask = []
        preds = []
        for i in chunked_data:
            i["labels"] = [i["labels"]]
            out = sdec_datamodule.tokenize_and_align_labels_inference(i)
            input_ids.append(out["input_ids"])
            p_labels.append(torch.as_tensor(out["labels"]))
            attention_mask.append(out["attention_mask"])

        for i, a, p in list(zip(input_ids, attention_mask, p_labels)):
            with torch.no_grad():
                embeddings = sdec_model.get_embeddings(i, a)
                fused = sdec_model.reconcile_features_labels(embeddings, p)
                outputs = sdec_model.forward(fused)
                preds.append(outputs[1].argmax(dim=-1))
            
        return preds
    
    def score(self, preds, labels):
        count = 0
        total = preds[0].squeeze().size()[0]
        for p_label, r_label in list(zip(preds[0].squeeze().tolist(), labels)):
            if p_label == r_label:
                count += 1

        return count/total


    # # # FUNCTIONS FOR TRANSCRIPTION # # # 

    def transcribe_audio_file(self, audio_file) -> None:
        """
        Transcribe an audio file with IBM Watson speech-to-text.
        """
        transcribe(audio_file)
    
    def load_watson_results(self, output_path):
        """
        Load Watson results from a json file.
        """
        return read_watson(output_path)
    
    def load_reference_from_txt(self, path_to_corrected_labels):
        """
        Load corrected labels from a text file.
        """
        return load_labels(path_to_corrected_labels)
    
    def save_watson_txt(self, watson_output, filepath):
        """
        Save results from Watson (dict) in a text file.
        """
        save_as_txt(watson_output, filepath)
        
        


