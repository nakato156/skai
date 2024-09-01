from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..gpt import gpt2_fine_tune
from pathlib import Path
from pprint import pprint
from .Reward import RewardModel

def preprocess_function(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 512, "return_tensors": "pt"}

    input_ids_chosen = []
    attention_mask_chosen = []
    input_ids_rejected = []
    attention_mask_rejected = []

    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["response"]):
        tokens_chosen = tokenizer.encode_plus(prompt + ": " + chosen, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt + ": " + rejected, **kwargs)

        input_ids_chosen.append(tokens_chosen["input_ids"].squeeze(0))
        attention_mask_chosen.append(tokens_chosen["attention_mask"].squeeze(0))
        input_ids_rejected.append(tokens_rejected["input_ids"].squeeze(0))
        attention_mask_rejected.append(tokens_rejected["attention_mask"].squeeze(0))

    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected,
    }


def generate_response(path_model, file_input_prompt):
    import pandas as pd

    df = pd.read_csv(file_input_prompt)
    responses = []
    for prompt in df["prompt"]: 
        prompt = prompt.replace(".", "")
        response = gpt2_fine_tune.generate_text(prompt, max_length=80, model_path=path_model).replace(prompt, "").replace(".:", "").strip(".").strip()
        
        # print(response)
        if "\n" in response: response = response.split("\n")[0]
        elif "." in response: response = response.split(".")[0]
        elif ":" in response: response = response.split(":")[0]

        pprint({"prompt": prompt, "response":response})
        responses.append(response)
    
    df["response"] = responses
    df.to_csv("prompts_for_score.csv", index=False)


if __name__ == '__main__':
    path_datasets = Path(__file__).parent.parent / "datasets" / "rlhf"
    dataset = load_dataset('csv', data_files= str(path_datasets / 'prompts_for_score.csv'))
    
    model_path="distilbert-base-multilingual-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=4, shuffle=True)

    reward_model = RewardModel(model=model, tokenizer=tokenizer, train_dataset=tokenized_dataset['train'])
    reward_model.train()

