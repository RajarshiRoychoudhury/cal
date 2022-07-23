import argparse
import logging
import os
from collections import defaultdict
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from project.src.estimators.mlp_estimator import MLP
from project.src.utils.data_loader import DatasetMapper, DatasetMapper2
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels, tokenizer):
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        encodings = self.text[idx]
        #print(encodings["input_ids"].shape)
        item = {"input_ids": torch.tensor(encodings["input_ids"].squeeze(0))}
        item["attention_mask"] = torch.tensor(encodings["attention_mask"])
        item['labels'] = torch.tensor(self.labels[idx])
        return item["input_ids"],  item["attention_mask"], item["labels"]

    def __len__(self):
        return len(self.labels)

class Dataset2(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        encodings = self.text[idx]
        #print(encodings["input_ids"].shape)
        item = {"input_ids": torch.tensor(encodings["input_ids"].squeeze(0))}
        item["attention_mask"] = torch.tensor(encodings["attention_mask"])
        return item["input_ids"],  item["attention_mask"]

    def __len__(self):
        return len(self.text)



class MLPEstimator:
    def __init__(self, args: argparse.Namespace,
                 vocab_size: int,
                 emb_dim: int,
                 num_labels: int,
#                  model,
#                  tokenizer):
                 pt_emb: np.ndarray) -> None:
        self.model = MLP(args, vocab_size, emb_dim, num_labels, pt_emb).to(DEVICE)
        #self.model = Model(args, vocab_size, emb_dim, num_labels, pt_emb).to(DEVICE)
#         self.model = model
#         self.model.to(DEVICE)
#         self.tokenizer = tokenizer
        self.criterion = nn.NLLLoss(reduction="mean")
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.args = args

        # cartography
        self.cartography_plot = {"correctness": [], "variability": [], "confidence": []}
        self.probabilities = defaultdict(list)
        self.correctness = defaultdict(list)
        self.gold_labels = defaultdict(list)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        #train = Dataset(X_train, y_train, self.tokenizer)
        train = DatasetMapper(torch.from_numpy(X_train).long(), torch.from_numpy(y_train).long())
        loader_train = DataLoader(train, batch_size=self.args.batch_size)
        representations = []

        # training
        self.model.train()

        for epoch in range(int(os.getenv("EPOCHS"))):
            epoch_loss = 0
            y_pred, y_gold = [], []

            for batch_x, batch_y in loader_train:
                self.model.zero_grad()
#                 print(batch_x)
#                 print(mask)
#                 print(batch_y)
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                #print(batch_x)
                if (
                        self.args.acquisition == "discriminative" or self.args.acquisition == "cartography") and epoch == int(
                        os.getenv("EPOCHS")) - 1:
                    raw_logits = self.model.forward_discriminative(batch_x)
                    for raw_logit in raw_logits:
                        representations.append(raw_logit.detach().cpu().numpy())

                raw_logits = self.model.forward(batch_x)
                predictions = self.model.predict_class(raw_logits)

                y_pred.extend(predictions)
                y_gold.extend(batch_y.tolist())
                # get probabilities and correctness per batch, for now only gold
                for idx, (raw_logit, gold_cls, pred_class) in enumerate(zip(raw_logits, batch_y, predictions),
                                                                        start=len(y_gold) - len(batch_x)):
                    self.probabilities[idx].append(float(torch.exp(raw_logit)[gold_cls]))
                    self.gold_labels[idx].append(int(gold_cls))
                    if gold_cls == pred_class:
                        self.correctness[idx].append(1)
                    else:
                        self.correctness[idx].append(0)

                loss = self.criterion(raw_logits, batch_y)
                loss.backward()

                self.optimizer.step()
                epoch_loss += loss

            logging.info(f"Estimator Epoch {epoch}: train loss: {epoch_loss / len(y_gold)} "
                          f"accuracy: {round(accuracy_score(y_pred, y_gold), 4)}")

        if self.args.acquisition == "discriminative" or self.args.acquisition == "cartography":
            return representations

    def predict(self, X_pool: np.ndarray) -> list:
        #pool = Dataset2(X_pool, self.tokenizer)
        pool = DatasetMapper2(torch.from_numpy(X_pool).long())
        loader_pool = DataLoader(pool, batch_size=self.args.batch_size)
        probas = []

        self.model.eval()
        for batch_x in loader_pool:
            batch_x = batch_x.to(DEVICE)

            with torch.no_grad():
                if self.args.acquisition == "bald":
                    raw_logits_list = [self.model.forward(batch_x) for _ in range(10)]
                    raw_logits_stacked = torch.stack(raw_logits_list).mean(dim=0).to(DEVICE)
                    probas.extend(self.model.predict_proba(raw_logits_stacked))

                elif self.args.acquisition == "discriminative" or self.args.acquisition == "cartography":
                    raw_logits = self.model.forward_discriminative(batch_x)
                    probas.extend(raw_logits.detach().cpu().numpy())

                else:
                    raw_logits = self.model.forward(batch_x)
                    probas.extend(self.model.predict_proba(raw_logits))

        return probas

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        #test = Dataset(X_test, y_test. self.tokenizer)
        test = DatasetMapper(torch.from_numpy(X_test).long(), torch.from_numpy(y_test).long())
        loader_test = DataLoader(test, batch_size=self.args.batch_size)

        y_pred = []
        y_gold = []

        self.model.eval()
        for batch_x, batch_y in loader_test:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            with torch.no_grad():
                raw_logits = self.model.forward(batch_x)
                predictions = self.model.predict_class(raw_logits)

                y_pred.extend(predictions)
                y_gold.extend(batch_y.tolist())

        test_accuracy = round(accuracy_score(y_pred, y_test), 4)

        return test_accuracy

    def weight_reset(self) -> None:
        self.model.weight_reset()

    def predict_class(self, predictions: torch.Tensor) -> List:
        return self.model.predict_class(predictions)
