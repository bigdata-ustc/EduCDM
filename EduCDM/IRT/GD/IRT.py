# coding: utf-8
# 2024/1/3 @ Gao Weibo

import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List
from EduCDM import CDM, re_index


class Net(nn.Module):

    def __init__(self, student_num: int, exer_n: int, value_range: int, a_range: int):
        super(Net, self).__init__()
        self.student_num = student_num
        self.exer_n = exer_n
        self.value_range = value_range
        self.a_range = a_range
        self.theta = nn.Embedding(self.student_num, 1)
        self.a = nn.Embedding(self.exer_n, 1)
        self.b = nn.Embedding(self.exer_n, 1)
        self.c = nn.Embedding(self.exer_n, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id: torch.Tensor, input_exercise: torch.Tensor) -> torch.Tensor:
        # before prednet
        theta = torch.squeeze(self.theta(stu_id), dim=-1)
        a = torch.squeeze(self.a(input_exercise), dim=-1)
        b = torch.squeeze(self.b(input_exercise), dim=-1)
        c = torch.squeeze(self.c(input_exercise), dim=-1)
        c = torch.sigmoid(c)
        theta = self.value_range * (torch.sigmoid(theta) - 0.5)
        b = self.value_range * (torch.sigmoid(b) - 0.5)
        if self.a_range != 0:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        # prednet using irt3pl
        return c + (1 - c) / (1 + torch.exp(-1.702 * a * (theta - b)))


class IRT(CDM):
    r'''
    The IRT model optimized by gradient descent.

    Args:
        meta_data: a dictionary containing all the userIds, and itemIds.

    Examples:
        meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5']}

        model = GDIRT(meta_data)
    '''

    def __init__(self, meta_data: dict):
        super(IRT, self).__init__()
        self.id_reindex, _ = re_index(meta_data)
        self.student_n = len(self.id_reindex['userId'])
        self.exer_n = len(self.id_reindex['itemId'])
        value_range = 1
        a_range = 10
        self.irt_net = Net(self.student_n, self.exer_n, value_range, a_range)

    def transform__(self, df_data: pd.DataFrame, batch_size: int, shuffle):
        users = [self.id_reindex['userId'][userId] for userId in df_data['userId'].values]
        items = [self.id_reindex['itemId'][itemId] for itemId in df_data['itemId'].values]
        responses = df_data['response'].values

        data_set = TensorDataset(
            torch.tensor(users, dtype=torch.int64),
            torch.tensor(items, dtype=torch.int64),
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
            device: device on which the model is trained. Default: 'cpu'. If you want to run it on your
                    GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.
            lr: learning rate. Default: 0.002.
            batch_size: the batch size during the training.
        '''

        self.irt_net = self.irt_net.to(device)
        self.irt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.irt_net.parameters(), lr=lr)
        train_data = self.transform__(train_data, batch_size, shuffle=True)
        for epoch_i in range(epoch):
            self.irt_net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.irt_net(user_id, item_id)
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

        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        test_loader = self.transform__(test_data, batch_size=64, shuffle=False)
        pred_proba = []
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Predicting"):
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                pred: torch.Tensor = self.irt_net(user_id, item_id)
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

    def save(self, filepath: str):
        r'''
        Save the model. This method is implemented based on the PyTorch's torch.save() method. Only the parameters in self.irt_net will be saved. You can save the whole IRT object using pickle.

        Args:
            filepath: the path to save the model.
        '''

        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath: str):
        r'''
        Load the model. This method loads the model saved at filepath into self.irt_net. Before loading, the object needs to be properly initialized.

        Args:
            filepath: the path from which to load the model.
        '''

        self.irt_net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
