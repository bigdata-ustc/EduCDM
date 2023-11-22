# coding: utf-8
# 2023/11/22 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from EduCDM import CDM, re_index


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, exer_n: int, student_n: int, knowledge_n: int, mf_type: str, dim: int, layer_dim1: int, layer_dim2: int):
        self.knowledge_n = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.emb_dim = dim
        self.mf_type = mf_type
        self.prednet_input_len = self.knowledge_n
        self.prednet_len1, self.prednet_len2 = layer_dim1, layer_dim2

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.student_n, self.emb_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.emb_dim)
        self.knowledge_emb = nn.Parameter(torch.zeros(self.knowledge_n, self.emb_dim))
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        if mf_type == 'gmf':
            self.k_diff_full = nn.Linear(self.emb_dim, 1)
            self.stat_full = nn.Linear(self.emb_dim, 1)
        elif mf_type == 'ncf1':
            self.k_diff_full = nn.Linear(2 * self.emb_dim, 1)
            self.stat_full = nn.Linear(2 * self.emb_dim, 1)
        elif mf_type == 'ncf2':
            self.k_diff_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.k_diff_full2 = nn.Linear(self.emb_dim, 1)
            self.stat_full1 = nn.Linear(2 * self.emb_dim, self.emb_dim)
            self.stat_full2 = nn.Linear(self.emb_dim, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.xavier_normal_(self.knowledge_emb)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        exer_emb = self.exercise_emb(input_exercise)
        # get knowledge proficiency
        batch, dim = stu_emb.size()
        stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        knowledge_emb = self.knowledge_emb.repeat(batch, 1).view(batch, self.knowledge_n, -1)
        if self.mf_type == 'mf':  # matrix factorization, simply inner product
            stat_emb = torch.sigmoid((stu_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':  # generalized matrix factorization
            stat_emb = torch.sigmoid(self.stat_full(stu_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':  # one-layer NCF
            stat_emb = torch.sigmoid(self.stat_full(torch.cat((stu_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':  # two-layer NCF
            stat_emb = torch.sigmoid(self.stat_full1(torch.cat((stu_emb, knowledge_emb), dim=-1)))
            stat_emb = torch.sigmoid(self.stat_full2(stat_emb)).view(batch, -1)
        batch, dim = exer_emb.size()
        exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.knowledge_n, 1)
        if self.mf_type == 'mf':
            k_difficulty = torch.sigmoid((exer_emb * knowledge_emb).sum(dim=-1, keepdim=False))  # batch, knowledge_n
        elif self.mf_type == 'gmf':
            k_difficulty = torch.sigmoid(self.k_diff_full(exer_emb * knowledge_emb)).view(batch, -1)
        elif self.mf_type == 'ncf1':
            k_difficulty = torch.sigmoid(self.k_diff_full(torch.cat((exer_emb, knowledge_emb), dim=-1))).view(batch, -1)
        elif self.mf_type == 'ncf2':
            k_difficulty = torch.sigmoid(self.k_diff_full1(torch.cat((exer_emb, knowledge_emb), dim=-1)))
            k_difficulty = torch.sigmoid(self.k_diff_full2(k_difficulty)).view(batch, -1)
        # get exercise discrimination
        e_discrimination = torch.sigmoid(self.e_discrimination(input_exercise))

        # prednet
        input_x = e_discrimination * (stat_emb - k_difficulty) * input_knowledge_point
        # f = input_x[input_knowledge_point == 1]
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class KaNCD(CDM):
    r'''
    The KaNCD model.
    Args:
        meta_data: a dictionary containing all the userIds, itemIds, and skills.
        dim: the dimension of the latent vectors of users, items and skills. default: 40
        mf_type: the type of layer(s) to be used for the interaction among latent vectors. default: "gmf"
            - "mf": matrix factorization, just inner product
            - "gmf": general matrix factorization. This is the layer used in the paper.
            - "ncf1": neural collaborative filtering, one layer.
            - "ncf2": neural collaborative filtering, two layers.
        layer_dim1: the dimension of the first hidden layer. Default: 512
        layer_dim2: the dimension of the second hidden layer. Default: 256

    Examples::
        meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}
        model = KaNCD(meta_data, 40, 'gmf', 512, 256)
    '''
    def __init__(self, meta_data: dict, dim: int=40, mf_type: str='gmf', layer_dim1: int=512, layer_dim2: int=256):
        super(KaNCD, self).__init__()
        assert mf_type in ['mf', 'gmf', 'ncf1', 'ncf2']
        self.id_reindex, _ = re_index(meta_data)
        self.student_n = len(self.id_reindex['userId'])
        self.exer_n = len(self.id_reindex['itemId'])
        self.knowledge_n = len(self.id_reindex['skill'])
        self.net = Net(self.exer_n, self.student_n, self.knowledge_n, mf_type, dim,  layer_dim1, layer_dim2)

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
            device: device on which the model is trained. Default: 'cpu'. If you want to run it on your
                    GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.
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

    def save(self, filepath):
        torch.save(self.net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=lambda s, loc: s))
        logging.info("load parameters from %s" % filepath)
