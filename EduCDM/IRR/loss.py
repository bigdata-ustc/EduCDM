# coding: utf-8
# 2021/6/19 @ tongshiwei

import torch
from torch import nn


def loss_mask(loss_list, n_samples):
    return [(i <= n_samples) * loss for i, loss in enumerate(loss_list)]


class PairSCELoss(nn.Module):
    def __init__(self):
        super(PairSCELoss, self).__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, pred1, pred2, sign=1, *args):
        """
        sign is either 1 or -1
        could be seen as predicting the sign based on the pred1 and pred2
        1: pred1 should be greater than pred2
        -1: otherwise
        """
        pred = torch.stack([pred1, pred2], dim=1)
        return self._loss(pred, ((torch.ones(pred1.shape[0], device=pred.device) - sign) / 2).long())


class HarmonicLoss(object):
    def __init__(self, zeta: (int, float) = 0.):
        self.zeta = zeta

    def __call__(self, point_wise_loss, pair_pred_loss, *args, **kwargs):
        return ((1 - self.zeta) * point_wise_loss + self.zeta * pair_pred_loss).mean()
