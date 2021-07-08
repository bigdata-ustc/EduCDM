# coding: utf-8
# 2021/3/23 @ tongshiwei

import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score


class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

    def forward(self, user_id, item_id):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)
        return torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)


class MCD(CDM):
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MCD, self).__init__()
        self.mf_net = MFNet(user_num, item_num, latent_dim)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.mf_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.mf_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.mf_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.mf_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.mf_net.train()
        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.mf_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.mf_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
