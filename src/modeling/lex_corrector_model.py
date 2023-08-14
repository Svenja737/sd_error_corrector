from transformers import DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from data_libs.metrics import compute_metrics
from data_libs.switchboard_prep import SwitchboardDataset

class LexSDCorrector:

    label_names = ["A", "B"]

    def __init__(self,
                 tokenizer_model="roberta-base",
                 model_name="roberta-for-token-classification"
                 ):
        
        self.compute_metrics = compute_metrics
        self.tokenizer_model = tokenizer_model
        self.model = model_name
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()}


    def prepare_data(self):

        data_loader = SwitchboardDataset("switchboard", "isip-aligned")
        train, val, test = data_loader.run_data_preparation()
        return train, val, test


    def set_train_args(self):

        args = TrainingArguments(
            self.model,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=10,
            weight_decay=0.01,
            push_to_hub=False,
        )

        return args


    def train_model(self, tokenized_train, tokenized_val):

        args = self.set_train_args()
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model, add_prefix_space=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(
            self.tokenizer_model,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=tokenizer
        )

        trainer.train()


    def test_model(self, tokenized_test):
        pass

    