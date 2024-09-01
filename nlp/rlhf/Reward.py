from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from trl import RewardTrainer

class RewardModel():
    def __init__(self, model, tokenizer, train_dataset, output_dir=None) -> None:
        self.output_dir = output_dir or "./reward_model"

        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.train_dataset = train_dataset
        
        self.peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Tarea de clasificación de secuencias
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            modules_to_save=["scores"],
            target_modules = 'all-linear',
        )

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,  # Directorio donde se guardarán los resultados
            per_device_train_batch_size=4,  # Tamaño del batch
            num_train_epochs=10, 
            optim="adamw_torch",
            save_steps=50,
            logging_dir="./logs",  # Directorio para los logs
            logging_steps=5,  # Cada cuántos pasos se realiza el logging
            report_to='tensorboard'
        )

    def train(self, save=True):
        trainer = RewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=self.train_dataset,
            peft_config=self.peft_config,
            max_length=120,
        )

        trainer.train()
        if save:
            trainer.model.save_pretrained(self.output_dir+"/final")
            self.tokenizer.save_pretrained(self.output_dir+"/final")