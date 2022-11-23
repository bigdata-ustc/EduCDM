# coding: utf-8

from tqdm import tqdm
import torch
from torch import nn
from baize.torch import loss_dict2tmt_torch_loss
from longling.ML.PytorchHelper import set_device
from longling.ML.PytorchHelper.toolkit.trainer import collect_params

from .ncd import NCDMNet
from .mirt import MIRTNet
from .dtn import DTN


class ICD(nn.Module):
    def __init__(self, user_n, item_n, know_n, cdm="ncd"):
        super(ICD, self).__init__()
        self.l_dtn = DTN(2 * item_n + 1, know_n)
        self.i_dtn = DTN(2 * user_n + 1, know_n)
        self.cdm_name = cdm
        if cdm == "ncd":
            self.cdm = NCDMNet(know_n, know_n)
        elif cdm == "mirt":
            self.cdm = MIRTNet(know_n)
        else:  # pragma: no cover
            raise ValueError()

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, u2i, u_mask, i2u, i_mask, i2k):
        u_trait = self.l_dtn(u2i, u_mask)
        v_trait = self.i_dtn(i2u, i_mask)
        return self.cdm(u_trait, v_trait, i2k)

    def get_user_profiles(self, batches):
        device = next(self.parameters()).device
        ids = []
        traits = []
        for _id, records, r_mask in tqdm(batches, "getting user profiles"):
            ids.append(_id.to("cpu"))
            traits.append(
                self.cdm.u_theta(
                    self.l_dtn(records.to(device),
                               r_mask.to(device))).to("cpu"))

        obj = {"uid": torch.cat(ids), "u_trait": torch.cat(traits)}
        return obj

    def get_item_profiles(self, batches):
        device = next(self.parameters()).device
        ids = []
        a = []
        b = []
        for _id, records, r_mask in tqdm(batches, "getting item profiles"):
            v_trait = self.i_dtn(records.to(device), r_mask.to(device))
            ids.append(_id.cpu())
            a.append(self.cdm.i_discrimination(v_trait).to("cpu"))
            b.append(self.cdm.i_difficulty(v_trait).to("cpu"))
        obj = {"iid": torch.cat(ids), "ia": torch.cat(a), "ib": torch.cat(b)}
        return obj


class DualICD(nn.Module):
    def __init__(self, stat_net: ICD, net: ICD, alpha=0.999):
        super(DualICD, self).__init__()
        self.stat_net = stat_net
        self.net = net
        self.alpha = alpha

    def momentum_weight_update(self, pre_net, train_select=None):
        """
        Momentum update of ICD
        """
        pre_net_params = collect_params(pre_net, train_select)
        net_params = collect_params(self.net, train_select)
        for param_pre, param_now in zip(pre_net_params, net_params):
            param_now.data = param_pre.data * self.alpha + param_now.data * (
                1. - self.alpha)

    def forward(self, u2i, u_mask, i2u, i_mask, i2k):
        output, theta, a, b = self.net(u2i, u_mask, i2u, i_mask, i2k)
        _, stat_theta, stat_a, stat_b = self.stat_net(u2i, u_mask, i2u, i_mask,
                                                      i2k)
        return output, theta, a, b, stat_theta, stat_a, stat_b


class EmbICD(nn.Module):
    def __init__(self, int_fc, weights):
        super(EmbICD, self).__init__()
        self.theta_emb = nn.Embedding(*weights[0].size(), _weight=weights[0])
        self.a_emb = nn.Embedding(*weights[1].size(), _weight=weights[1])
        if len(weights[2].size()) == 1:
            self.b_emb = nn.Embedding(weights[2].size()[0],
                                      1,
                                      _weight=torch.unsqueeze(weights[2], 1))
        else:
            self.b_emb = nn.Embedding(*weights[2].size(), _weight=weights[2])
        self.int_fc = int_fc
        self._user_id2idx = {}
        self._item_id2idx = {}

    def build_user_id2idx(self, users):
        idx = 0
        for user_id in users:
            if user_id not in self._user_id2idx:
                self._user_id2idx[user_id] = idx
                idx += 1

    def build_item_id2idx(self, items):
        idx = 0
        for item_id in items:
            if item_id not in self._item_id2idx:
                self._item_id2idx[item_id] = idx
                idx += 1

    def user_id2idx(self, users):
        users_idx = []
        for user in users:
            users_idx.append(self._user_id2idx[user])
        return users_idx

    def item_id2idx(self, items):
        items_idx = []
        for item in items:
            items_idx.append(self._item_id2idx[item])
        return items_idx

    def forward(self, user_idx, item_idx, know):
        theta = self.theta_emb(user_idx).detach()
        a = self.a_emb(item_idx).detach()
        b = self.b_emb(item_idx).detach()

        theta.requires_grad_(True)
        a.requires_grad_(True)
        b.requires_grad_(True)

        return self.int_fc(theta, a, torch.squeeze(b),
                           know).view(-1), theta, a, b


class DeltaTraitLoss(nn.Module):
    def __init__(self):
        super(DeltaTraitLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, theta, a, b, stat_theta, stat_a, stat_b):
        return self.mse_loss(theta, stat_theta) + self.mse_loss(
            a, stat_a) + self.mse_loss(b, stat_b)


class DualLoss(nn.Module):
    def __init__(self, beta=0.95, *args, **kwargs):
        super(DualLoss, self).__init__()
        self.beta = beta
        self.bce = nn.BCELoss(*args, **kwargs)
        self.delta_trait = DeltaTraitLoss()

    def forward(self, pred, truth, theta, a, b, stat_theta, stat_a, stat_b):
        return self.beta * self.bce(
            pred, truth) + (1. - self.beta) * self.delta_trait(
                theta, a, b, stat_theta, stat_a, stat_b)


def get_dual_loss(ctx, beta=0.95, *args, **kwargs):
    return loss_dict2tmt_torch_loss({
        "Loss":
        set_device(DualLoss(beta, *args, **kwargs), ctx),
        "BCE":
        set_device(torch.nn.BCELoss(*args, **kwargs), ctx),
        "DTL":
        set_device(DeltaTraitLoss(), ctx),
    })


def get_loss(ctx, *args, **kwargs):  # pragma: no cover
    return loss_dict2tmt_torch_loss(
        {"cross entropy": set_device(torch.nn.BCELoss(*args, **kwargs), ctx)})


def get_net(ctx=None, *args, **kwargs):
    if ctx is None:  # pragma: no cover
        return ICD(*args, **kwargs)
    return set_device(ICD(*args, **kwargs), ctx)
