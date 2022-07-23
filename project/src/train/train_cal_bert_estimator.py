import argparse
import logging
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from project.src.estimators.dal_estimator import DALMLP
from project.src.utils.data_loader import DatasetDiscriminativeBERT, DatasetDiscriminativeBERT2, DatasetMapperDiscriminative, DatasetMapperDiscriminative2
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


class BERTCALEstimator:
    def __init__(self, args: argparse.Namespace, model) -> None:
        self.model = model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        self.args = args

    def train(self, X_train: np.ndarray, y_train: np.ndarray):

        train = DatasetMapperDiscriminative(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader_train = DataLoader(train, batch_size=self.args.batch_size)
        self.model.train()
        
        for epoch in range(int(os.getenv("EPOCHS_DISCRIMINATIVE"))):
            epoch_loss = 0
            y_pred = []
            y_gold = []
            #print(next(iter(loader_train)))
            for batch_x , batch_y, idx in loader_train:
                self.model.zero_grad()

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                raw_logits = self.model.forward(batch_x)
                predictions = self.model.predict_class(raw_logits)

                y_pred.extend(predictions)
                y_gold.extend(batch_y.tolist())

                loss = self.criterion(raw_logits, batch_y)
                loss.backward()

                self.optimizer.step()
                epoch_loss += loss

            logger.info(f"BERTCALESTIMATOR Epoch {epoch}: train loss: {epoch_loss / len(y_gold)} "
                         f"accuracy: {round(accuracy_score(y_pred, y_gold), 4)}")

    def predict(self, X_pool: np.ndarray) -> list:
        pool = DatasetMapperDiscriminative2(torch.from_numpy(X_pool))
        loader_pool = DataLoader(pool, batch_size=self.args.batch_size)
        probas = []
        indices = []
        #classes = []
        self.model.eval()
        for batch_x, idx in loader_pool:
            batch_x = batch_x.to(DEVICE)
            indices.extend(idx)

            with torch.no_grad():
                raw_logits = self.model.forward(batch_x)
                #classes.extend(self.model.predict_class(batch_x))
                probas.extend(self.model.predict_proba(raw_logits))

        idx_with_probas = [(idx, proba) for idx, proba in zip(indices, probas)]
        idx_with_probas.sort(key=lambda tup: (torch.amax(tup[1])))
        top_k_indices = [idx.item() for idx, proba in idx_with_probas[:int(os.getenv("ACTIVE_LEARNING_BATCHES"))]]
#         with open("predicted.txt", "w") as f:
#             for i in classes:
#                 f.write(str(i))
#                 f.write("\n")
        return top_k_indices

    def weight_reset(self) -> None:
        self.model.weight_reset()

    def predict_class(self, predictions: torch.Tensor) -> List:
        return self.model.predict_class(predictions)
