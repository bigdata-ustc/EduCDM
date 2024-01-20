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


class MIRTNet(nn.Module):

    def __init__(self, student_n: int, exer_n: int, latent_dim: int, a_range: int):
        self.user_num = student_n
        self.item_num = exer_n
        self.latent_dim = latent_dim
        self.a_range = a_range

        super(MIRTNet, self).__init__()

        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id: torch.Tensor, input_exercise: torch.Tensor) -> torch.Tensor:
        theta = torch.squeeze(self.theta(stu_id), dim=-1)
        a = torch.squeeze(self.a(input_exercise), dim=-1)
        a = self.a_range * torch.sigmoid(a)
        b = torch.squeeze(self.b(input_exercise), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return 1 / (1 + torch.exp(- torch.sum(torch.multiply(a, theta), axis=-1) + b))


class MIRT(CDM):
    r'''
    The MIRT model.

    Args:
        meta_data: a dictionary containing all the userIds, itemIds.
        latent_dim: the dimension of user embedding. Default: 20
        [a_range]: the influence degree of item discrimination. Default: 1

    Examples:
        meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5']}
        model = MIRT(meta_data, 20)

    '''

    def __init__(self, meta_data: dict, latent_dim=20, a_range=1):
        super(MIRT, self).__init__()
        self.id_reindex, _ = re_index(meta_data)
        self.student_n = len(self.id_reindex['userId'])
        self.exer_n = len(self.id_reindex['itemId'])
        self.mirt_net = MIRTNet(self.student_n, self.exer_n, latent_dim, a_range)

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

        self.mirt_net = self.mirt_net.to(device)
        self.mirt_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.mirt_net.parameters(), lr=lr)
        train_data = self.transform__(train_data, batch_size, shuffle=True)
        for epoch_i in range(epoch):
            self.mirt_net.train()
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.mirt_net(user_id, item_id)
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

        self.mirt_net = self.mirt_net.to(device)
        self.mirt_net.eval()
        test_loader = self.transform__(test_data, batch_size=64, shuffle=False)
        pred_proba = []
        with torch.no_grad():
            for batch_data in tqdm(test_loader, "Predicting"):
                user_id, item_id, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                pred: torch.Tensor = self.mirt_net(user_id, item_id)
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
        df_pred = pd.DataFrame({'userId': df_proba['userId'], 'itemId': df_proba['itemId'], 'proba': y_pred})

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
        Save the model. This method is implemented based on the PyTorch's torch.save() method. Only the parameters
        in self.mirt_net will be saved. You can save the whole NCDM object using pickle.

        Args:
            filepath: the path to save the model.
        '''

        torch.save(self.mirt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath: str):
        r'''
        Load the model. This method loads the model saved at filepath into self.mirt_net. Before loading, the object
        needs to be properly initialized.

        Args:
            filepath: the path from which to load the model.

        Examples:

        '''

        self.mirt_net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
