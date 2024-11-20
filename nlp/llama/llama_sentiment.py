import os
import pandas as pd
from datasets import Dataset
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-sentiment"

def process_df(path):
    df = pd.read_csv(path)
    dataset = pd.DataFrame({
        'train': [('<s>[INS] ' + row['text'] + '[/INS]' + row['response'] + ' <s>') for _, row in df.iterrows()]
    })
    return dataset

class LlamaSentimen():
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        
    def __configLora(self):
        ################################################################################
        # QLoRA parameters
        ################################################################################

        # LoRA attention dimension
        self.lora_r = 64
        # Alpha parameter for LoRA scaling
        self.lora_alpha = 16
        # Dropout probability for LoRA layers
        self.lora_dropout = 0.2

        ################################################################################
        # bitsandbytes parameters
        ################################################################################

        # Activate 4-bit precision base model loading
        self.use_4bit = True
        # Compute dtype for 4-bit base models
        self.bnb_4bit_compute_dtype = "float16"
        # Quantization type (fp4 or nf4)
        self.bnb_4bit_quant_type = "nf4"

        # Activate nested quantization for 4-bit base models (double quantization)
        self.use_nested_quant = False

    def train(self, num_train_epochs=4, output_dir="./results", learning_rate=2e-4, optim="paged_adamw_32bit"):
        self.__configLora()
        
        ################################################################################
        # TrainingArguments parameters
        ################################################################################

        # Enable fp16/bf16 training (set bf16 to True with an A100)
        fp16 = False
        bf16 = False

        # Batch size per GPU for training
        per_device_train_batch_size = 4
        # Batch size per GPU for evaluation
        per_device_eval_batch_size = 4
        # Number of update steps to accumulate the gradients for
        gradient_accumulation_steps = 1
        # Enable gradient checkpointing
        gradient_checkpointing = True
        # Maximum gradient normal (gradient clipping)
        max_grad_norm = 0.3        
        # Weight decay to apply to all layers except bias/LayerNorm weights
        weight_decay = 0.001
        
        # Learning rate schedule
        lr_scheduler_type = "cosine"
        # Number of training steps (overrides num_train_epochs)
        max_steps = -1
        # Ratio of steps for a linear warmup (from 0 to learning rate)
        warmup_ratio = 0.03

        # Group sequences into batches with same length
        # Saves memory and speeds up training considerably
        group_by_length = True
        # Save checkpoint every X updates steps
        save_steps = 0
        # Log every X updates steps
        logging_steps = 25

        ################################################################################
        # SFT parameters
        ################################################################################

        # Maximum sequence length to use
        max_seq_length = None
        # Pack multiple short examples in the same input sequence to increase efficiency
        packing = False
        # Load the entire model on the GPU 0
        device_map = {"": 0}

        dataset = Dataset.from_pandas(dataset)

        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and self.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        # Load LLaMA tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard"
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="train",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=packing,
        )

        self.model = model

        # Train model
        trainer.train()
        # Save trained model
        trainer.model.save_pretrained(new_model)

if __name__ == '__main__':
    dataset = process_df('train_dataset.csv')
    dataset.to_csv('train_dataset_llama.csv', index=False)