# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def build_text_files(df:pd.DataFrame, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for _, row in df.iterrows():
        if row["text"] == "none" or row["response"] == "none": continue
        data += row["text"] + ': ' + row["response"] + "\n"
    f.write(data)

def load_dataset(train_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=120)

    # test_dataset = TextDataset(
    #       tokenizer=tokenizer,
    #       file_path=test_path,
    #       block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,data_collator

def train_model(train_file_path,
        model_name,
        output_dir,
        overwrite_output_dir,
        per_device_train_batch_size,
        num_train_epochs,
        save_steps):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset, data_collator = load_dataset(train_file_path, tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
        )

    trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
    )
        
    trainer.train()
    trainer.save_model()

def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def generate_text(sequence, max_length, model_path="./my-gpt2-sentiment"):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        # num_beams=5,
        # no_repeat_ngram_size=2,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False)
    args = parser.parse_args()

    if args.train:
        data = pd.read_csv('../datasets/gpt/train_dataset.csv')
        data.sample(frac=1).reset_index(drop=True)
        print(data.shape)

        # train, test = train_test_split(data, test_size=0.15)

        build_text_files(data,'../datasets/gpt/train_resume.txt')
        # build_text_files(test,'../datasets/gpt/test_resume.txt')

        # print("Train dataset length: "+str(len(train)))
        # print("Test dataset length: "+ str(len(test)))


        # you need to set parameters 
        train_file_path = "../datasets/gpt/train_resume.txt"
        model_name = args.model_name
        output_dir = './my-gpt2-sentiment'
        overwrite_output_dir = args.overwrite_output_dir
        per_device_train_batch_size = 8
        num_train_epochs = 80
        save_steps = 500

        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        train_model(
            train_file_path=train_file_path,
            model_name=model_name,
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps
        )

    sequence = "No puedo creer que hayan cancelado mi serie favorita, estoy devastado: "
    max_len = 100
    print(generate_text(sequence, max_len))
