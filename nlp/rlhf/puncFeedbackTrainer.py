
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

MODEL_NAME = "bert-base-multilingual-cased"
N_CLASES = 5
RANDOM_SEED = 42
MAX_LEN = 180
DATASET_PATH = '../datasets/rlhf/feedback_prompt.csv'
BATCH_SIZE = 8 # 16

class FeedbackDataset(Dataset):
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
    ds = FeedbackDataset(
        texts=(df.prompt + ':' + df.response).to_numpy(),
        labels=df.score.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

class BertClasifyer(nn.Module):
    def __init__(self, n_classes):
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        super(BertClasifyer, self).__init__()
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

class PunctuationTrainer:
    def __init__(self, n_classes, device=None) -> None:
        self.device:str = device
        self.model = BertClasifyer(n_classes)
        self.model = self.model.to(self.device)

        self.__df_train = None
        self.__df_test = None
        self.__train_data_loader = None
        self.__test_data_loader = None
        self.tokenizer = None

    def _config_(self, max_len, epochs, batch_size, optimizer=None, scheduler=None, loss_fn=None):
        self.optimizer = optimizer or optim.AdamW(self.model.parameters(), lr=2e-5)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss().to(device)
        
        if self.tokenizer is None: self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        if self.__df_test is None or self.__df_train is None:
            df = pd.read_csv(DATASET_PATH)
            df.score = df.score.astype(int).apply(lambda x: x - 1)

            df_train, df_test = train_test_split(df, test_size=.2, random_state=RANDOM_SEED)
            
            self.__df_train = df_train
            self.__df_test = df_test
        
        if self.__train_data_loader is None or self.__test_data_loader is None:
            self.__train_data_loader = data_loader(self.__df_train, self.tokenizer, max_len, batch_size)
            self.__test_data_loader = data_loader(self.__df_test, self.tokenizer, max_len, batch_size)

        if scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(self.__train_data_loader) * epochs # total epochs
            )

    def __train_epoch(self, dataloader, optimizer, scheduler, n_examples):
        model = self.model.train()
        losses = []
        correct_predictions = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds.eq(labels)).item()
            loss = self.loss_fn(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions / n_examples, np.mean(losses)

    def eval_model(self, dataloader, loss_fn, device, n_examples):
        model = self.model.eval()
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

    def train(self, epochs, max_len, batch_size, optimizer=None, scheduler=None, loss_fn=None, save=True):
        self._config_(max_len, epochs, batch_size, optimizer, scheduler, loss_fn)
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            print('-' * 10)

            train_acc, train_loss = self.__train_epoch(self.model, self.__train_data_loader, self.optimizer, self.scheduler, len(self.__df_train))
            test_acc, test_loss = self.eval_model(self.model, self.__test_data_loader, self.loss_fn, device, len(self.__df_test))

            print(f'Train loss {train_loss}\taccuracy {train_acc}')
            print(f'Test loss {test_loss}\taccuracy {test_acc}')

        if save:
            from pathlib import Path

            path_models = Path('./models')
            if not path_models.exists(): 
                Path('./models').mkdir(parents=True, exist_ok=True)
                
            torch.save(model, path_models / 'reward_model.pth')

    def load(self, path='./models/reward_model.pth'):
        self.model = torch.load(path)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def scorePrompt(self, text: str) -> int:
        encoding = self.tokenizer.encode_plus(
                text,
                max_length=MAX_LEN,
                truncation=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
        device = self.device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = torch.max(output, dim=1)

        return pred.item()

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max_len', type=int, default=MAX_LEN)
    parser.add_argument('--n_classes', type=int, default=N_CLASES)
    parser.add_argument('--predict', type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PunctuationTrainer(n_classes=args.n_classes, device=device)

    if args.train:
        model.train(epochs=args.epochs, max_len=args.max_len, batch_size=args.batch_size)        
    else:
        model.load('./models/reward_model.pth')

    if args.predict:
        text = args.predict
    else:
        text = "Se me partía el alma al escuchar eso: Parece que estás sintiendo una profunda trsiteza por un suceso"
        print(text)

    cls = model.scorePrompt(text)
    print(cls)