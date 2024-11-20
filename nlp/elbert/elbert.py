# -*- coding: utf-8 -*-

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

RANDOM_SEED = 42
MAX_LEN = 180
BATCH_SIZE = 8 # 16
DATASET_PATH = '../datasets/elbert/training_es.csv'
N_CLASES = 6

MODEL_NAME = 'bert-base-multilingual-cased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.text = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            "text": text,
            "input_ids": encoding['input_ids'].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def data_loader(df, tokenizer, max_len, batch_size):
    ds = EmotionDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

class BertSentimentClasifyer(nn.Module):
    def label_to_text(self, label: int) -> str:
        return ("Tristeza", "Felicidad", "Amor", "Enojo", "Sorpresa")[label]

    def __init__(self, n_classes):
        super(BertSentimentClasifyer, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, cls_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output

def train_model(model, dataloader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds.eq(labels)).item()
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions / n_examples, np.mean(losses)

def eval_model(model, dataloader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)

            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def train(save=True):
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        test_acc, test_loss = eval_model(model, test_data_loader,  loss_fn, device, len(df_test))

        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Test loss {test_loss} accuracy {test_acc}')

    if save:
        from pathlib import Path

        path_models = Path('./models')
        if not path_models.exists(): 
            Path('./models').mkdir(parents=True, exist_ok=True)
            
        torch.save(model.state_dict(), path_models / 'ElBERT.pth')
        torch.save(model, path_models / 'modelo_completo.pth')

def load_weights(path) -> BertSentimentClasifyer:
    model = BertSentimentClasifyer(N_CLASES)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

def load_model(path) -> BertSentimentClasifyer:
    BertSentimentClasifyer(N_CLASES)
    if path == 'default':
        path = './models/v1.1/ElBERT.pth'
    return torch.load(path)

def clasifySentiment(model, text: str):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    encoding = tokenizer.encode_plus(
            text,
            max_length=MAX_LEN,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()  # Asegúrate de que el modelo está en modo evaluación
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = torch.max(output, dim=1)
    
    return pred.item()

if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    print(device)

    df = pd.read_csv(DATASET_PATH)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    df_train, df_test = train_test_split(df, test_size=.2, random_state=RANDOM_SEED)

    train_data_loader = data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    model = BertSentimentClasifyer(N_CLASES)
    model = model.to(device)

    epochs = 10
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    emociones = {
        0: "tristesa",
        1: "felicidad",
        2: "amor",
        3: "enojo",
        4: "miedo",
        5: "ni idea"
    }

    train(save=True)
    text = input(">>> ")
    cls = clasifySentiment(model, text)
    print(emociones[cls])