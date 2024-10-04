# coding: utf-8
# 2024/01/17 @ CSLiJT

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import CDM, re_index


class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)


class DINA(CDM):

    def __init__(self, meta_data: dict,
                 max_slip: float = 0.4,
                 max_guess: float = 0.4):
        super(DINA, self).__init__()
        self.id_reindex, _ = re_index(meta_data)
        self.student_n = len(self.id_reindex['userId'])
        self.exer_n = len(self.id_reindex['itemId'])
        self.knowledge_n = len(self.id_reindex['skill'])
        self.net = DINANet(
            self.student_n,
            self.exer_n,
            self.knowledge_n,
            max_slip=max_slip,
            max_guess=max_guess)

    def transform__(self, df_data: pd.DataFrame, batch_size: int, shuffle):
        users = [self.id_reindex['userId'][userId] for userId in df_data['userId'].values]
        items = [self.id_reindex['itemId'][itemId] for itemId in df_data['itemId'].values]
        responses = df_data['response'].values
        knowledge_emb = torch.zeros((len(df_data), self.knowledge_n))
        for idx, skills in enumerate(df_data['skill']):
            skills = eval(skills)  # str of list to list
            for skill in skills:
                skill_reindex = self.id_reindex['skill'][skill]
                knowledge_emb[idx][skill_reindex] = 1.0

        data_set = TensorDataset(
            torch.tensor(users, dtype=torch.int64),
            torch.tensor(items, dtype=torch.int64),
            knowledge_emb,
            torch.tensor(responses, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    def fit(self, train_data: pd.DataFrame, epoch: int, val_data=None, device="cpu", lr=0.002, batch_size=64):
        r'''
        Train the model with train_data. If val_data is provided, print the AUC and accuracy on val_data after each epoch.

        Args:
            train_data: a dataframe containing training userIds, itemIds and responses.
            epoch: number of training epochs.
            val_data: a dataframe containing validation userIds, itemIds and responses. Default: None.
            device: device on which the model is trained. Default: 'cpu'. If you want to run it on your GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.
            lr: learning rate. Default: 0.002.
            batch_size: the batch size during the training.
        '''
        self.net = self.net.to(device)
        self.net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)
        train_data = self.transform__(train_data, batch_size, shuffle=True)
        for epoch_i in range(epoch):
            self.net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if val_data is not None:
                auc, accuracy = self.eval(val_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (epoch_i, auc, accuracy))

    def predict_proba(self, test_data: pd.DataFrame, device="cpu") -> pd.DataFrame:
        r'''
        Output the predicted probabilities that the users would provide correct answers using test_data.
        The probabilities are within (0, 1).

        Args:
            test_data: a dataframe containing testing userIds and itemIds.
            device: device on which the model is trained. Default: 'cpu'. If you want to run it on your
                    GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.

        Return:
            a dataframe containing the userIds, itemIds, and proba (predicted probabilities).
        '''

        self.net = self.net.to(device)
        self.net.eval()
        test_loader = self.transform__(test_data, batch_size=64, shuffle=False)
        pred_proba = []
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Predicting"):
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                pred: torch.Tensor = self.net(user_id, item_id, knowledge_emb)
                pred_proba.extend(pred.detach().cpu().tolist())
        ret = pd.DataFrame({'userId': test_data['userId'], 'itemId': test_data['itemId'], 'proba': pred_proba})
        return ret

    def predict(self, test_data: pd.DataFrame, device="cpu") -> pd.DataFrame:
        r'''
        Output the predicted responses using test_data. The responses are either 0 or 1.

        Args:
            test_data: a dataframe containing testing userIds and itemIds.
            device: device on which the model is trained. Default: 'cpu'. If you want to run it on your
                    GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.

        Return:
            a dataframe containing the userIds, itemIds, and predicted responses.
        '''

        df_proba = self.predict_proba(test_data, device)
        y_pred = [1.0 if proba >= 0.5 else 0 for proba in df_proba['proba'].values]
        df_pred = pd.DataFrame({'userId': df_proba['userId'], 'itemId': df_proba['itemId'], 'pred': y_pred})

        return df_pred

    def eval(self, val_data: pd.DataFrame, device="cpu") -> Tuple[float, float]:
        r'''
        Output the AUC and accuracy using the val_data.

        Args:
            val_data: a dataframe containing testing userIds and itemIds.
            device: device on which the model is trained. Default: 'cpu'. If you want to run it on your
                    GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.

        Return:
            AUC, accuracy
        '''

        y_true = val_data['response'].values
        df_proba = self.predict_proba(val_data, device)
        pred_proba = df_proba['proba'].values
        return roc_auc_score(y_true, pred_proba), accuracy_score(y_true, np.array(pred_proba) >= 0.5)

    def save(self, filepath):
        r'''
        Save the model. This method is implemented based on the PyTorch's torch.save() method. Only the parameters
        in self.dina_net will be saved. You can save the whole DINA object using pickle.

        Args:
            filepath: the path to save the model.
        '''

        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        r'''
        Load the model. This method loads the model saved at filepath into self.dina_net. Before loading, the object
        needs to be properly initialized.

        Args:
            filepath: the path from which to load the model.

        Examples:
            model = DINA(meta_data)  # where meta_data is from the same dataset which is used to train the model at filepath
            model.load('path_to_the_pre-trained_model')
        '''

        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
