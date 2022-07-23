import argparse
import logging
import os
from collections import defaultdict
from typing import Any, List

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from project.src.estimators.mlp_estimator import MLP
from project.src.utils.data_loader import DatasetMapper, DatasetMapper2, Dataset, Dataset2
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


class BertEstimator:
    def __init__(self, args: argparse.Namespace,
                #  vocab_size: int,
                #  emb_dim: int,
                 model):
                 #pt_emb: np.ndarray) -> None:
        #self.model = Model(args, vocab_size, emb_dim, num_labels, pt_emb).to(DEVICE)
        self.model = model.to(DEVICE)
#         self.criterion = nn.NLLLoss(reduction="mean")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001)
        self.args = args

        # cartography
        self.cartography_plot = {"correctness": [], "variability": [], "confidence": []}
        self.probabilities = defaultdict(list)
        self.correctness = defaultdict(list)
        self.gold_labels = defaultdict(list)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        train = Dataset(X_train,y_train)
        loader_train = DataLoader(train, batch_size=self.args.batch_size)
        representations = []

        # training
        self.model.train()

        for epoch in range(int(os.getenv("EPOCHS"))):
            epoch_loss = 0
            y_pred, y_gold = [], []
            #print(next(iter(loader_train)))
            for input_ids, attentions_mask, batch_y in loader_train:
                self.model.zero_grad()

                input_ids = input_ids.to(DEVICE)
                attentions_mask = attentions_mask.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                if (
                        self.args.acquisition == "discriminative" or self.args.acquisition == "cartography") and epoch == int(
                        os.getenv("EPOCHS")) - 1:
                    raw_logits = self.model.forward_discriminative(input_ids, attentions_mask)
                    for raw_logit in raw_logits:
                        representations.append(raw_logit.detach().cpu().numpy())

                raw_logits = self.model.forward(input_ids, attentions_mask)
                predictions = self.model.predict_class(raw_logits)

                y_pred.extend(predictions)
                y_gold.extend(batch_y.tolist())
#                 print(y_pred)
#                 print(y_gold)
                # get probabilities and correctness per batch, for now only gold
                for idx, (raw_logit, gold_cls, pred_class) in enumerate(zip(raw_logits, batch_y, predictions),
                                                                        start=len(y_gold) - len(input_ids)):
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
        pool = Dataset2(X_pool)
        loader_pool = DataLoader(pool, batch_size=self.args.batch_size)
        probas = []

        self.model.eval()
        for input_ids, attentions_mask in loader_pool:
            input_ids = input_ids.to(DEVICE)
            attentions_mask = attentions_mask.to(DEVICE)

            with torch.no_grad():
                if self.args.acquisition == "bald":
                    raw_logits_list = [self.model.forward(input_ids, attentions_mask) for _ in range(10)]
                    raw_logits_stacked = torch.stack(raw_logits_list).mean(dim=0).to(DEVICE)
                    probas.extend(self.model.predict_proba(raw_logits_stacked))

                elif self.args.acquisition == "discriminative" or self.args.acquisition == "cartography":
                    raw_logits = self.model.forward_discriminative(input_ids, attentions_mask)
                    probas.extend(raw_logits.detach().cpu().numpy())

                else:
                    raw_logits = self.model.forward(input_ids, attentions_mask)
                    probas.extend(self.model.predict_proba(raw_logits))

        return probas

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        test = Dataset(X_test, y_test)
        loader_test = DataLoader(test, batch_size=self.args.batch_size)

        y_pred = []
        y_gold = []

        self.model.eval()
        for input_ids, attentions_mask, batch_y in loader_test:
            input_ids = input_ids.to(DEVICE)
            attentions_mask = attentions_mask.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            with torch.no_grad():
                raw_logits = self.model.forward(input_ids, attentions_mask)
                predictions = self.model.predict_class(raw_logits)
                y_pred.extend(predictions)
                y_gold.extend(batch_y.tolist())
        filePrefix = os.getenv("FILE_PREFIX")
        test_accuracy = round(accuracy_score(y_pred, y_test), 4)
        with open("predicted_{}.txt".format(filePrefix), "w") as f:
            for i in y_pred:
                f.write(str(i))
                f.write("\n")
        return test_accuracy
    
    def predict_class(self, X_pool, pool_path):
        pool = Dataset2(X_pool)
        loader_pool = DataLoader(pool, batch_size=self.args.batch_size)

        y_pred = []

        self.model.eval()
        for input_ids, attentions_mask in loader_pool:
            input_ids = input_ids.to(DEVICE)
            attentions_mask = attentions_mask.to(DEVICE)
            with torch.no_grad():
                raw_logits = self.model.forward(input_ids, attentions_mask)
                predictions = self.model.predict_class(raw_logits)
                y_pred.extend(predictions)
        filePrefix = os.getenv("FILE_PREFIX")
        X = list(pd.read_csv(pool_path)["Description"].values)
        with open("toxic_{}_predicted.txt".format(filePrefix), "w") as f1:
            with open("non_toxic_{}_predicted.txt".format(filePrefix), "w") as f2:
                for i in range(len(y_pred)):
                    if y_pred[i]==1:
                        f1.write(X[i])
                        f1.write("\n")
                    elif y_pred[i]==0:
                        f2.write(X[i])
                        f2.write("\n")
#         return test_accuracy

    def weight_reset(self) -> None:
        self.model.weight_reset()

#     def predict_class(self, predictions: torch.Tensor) -> List:
#         return self.model.predict_class(predictions)
