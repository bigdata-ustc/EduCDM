# coding: utf-8

import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from EduCDM.MIRT.MIRT import MIRTNet as _EmbedMIRTNet
from EduCDM.IRT.GD.IRT import IRTNet as _EmbedIRTNet
from EduCDM.DINA.GD.DINA import DINANet as _EmbedDINANet
from baize.torch import loss_dict2tmt_torch_loss, save_params
from longling.ML.PytorchHelper import set_device
from ICD.sym import PosLinear


def get_loss(ctx, *args, **kwargs):
    return loss_dict2tmt_torch_loss(
        {"cross entropy": set_device(torch.nn.BCELoss(*args, **kwargs), ctx)}
    )


class EmbedMIRTNet(_EmbedMIRTNet):
    def __init__(self, user_num, item_num, know_n, *args, **kwargs):
        super(EmbedMIRTNet, self).__init__(user_num, item_num, know_n, a_range=1, *args, **kwargs)

    def forward(self, user, item, *args):
        theta = self.u_theta(user)
        a = self.i_discrimination(item)
        b = self.i_difficulty(item)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(theta, a, b, **self.irf_kwargs)

    def get_user_profiles(self, user):
        user = torch.tensor(user, dtype=torch.int64).to(
            self.theta.weight.device) if not isinstance(user, torch.Tensor) else user
        obj = {
            "uid": user,
            "u_trait": self.u_theta(user)
        }
        return obj

    def save_user_profiles(self, user, filepath):
        obj = self.get_user_profiles(user)
        torch.save(obj, filepath)
        print("save user profiles to %s" % os.path.abspath(filepath))

    def get_item_profiles(self, item):
        item = torch.tensor(item, dtype=torch.int64).to(
            self.a.weight.device) if not isinstance(item, torch.Tensor) else item
        obj = {
            "iid": item,
            "ib": self.i_difficulty(item),
            "ia": self.i_discrimination(item),
        }
        return obj

    def save_item_profiles(self, item, filepath):
        obj = self.get_item_profiles(item)
        torch.save(obj, filepath)
        print("save item profiles to %s" % os.path.abspath(filepath))

    def u_theta(self, user):
        return (torch.sigmoid(torch.squeeze(self.theta(user), dim=-1)) - 0.5) * 6

    def i_difficulty(self, item):
        return (torch.sigmoid(torch.squeeze(self.b(item), dim=-1)) - 0.5) * 6

    def i_discrimination(self, item):
        a = torch.squeeze(self.a(item), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        return a

class EmbedIRTNet(_EmbedIRTNet):
    def __init__(self, user_num, item_num,know_n, value_range=6, a_range=1, irf_kwargs=None):
        super().__init__(user_num, item_num, value_range, a_range, irf_kwargs)

    def forward(self, user, item, *args):
        theta = self.u_theta(user)
        a = self.i_discrimination(item)
        b = self.i_difficulty(item)
        c = self.i_guess(item)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        # return self.irf(theta, a, b, **self.irf_kwargs)
        return self.irf(theta, a, b, c, **self.irf_kwargs)

    def get_user_profiles(self, user):
        user = torch.tensor(user, dtype=torch.int64).to(
            self.theta.weight.device) if not isinstance(user, torch.Tensor) else user
        obj = {
            "uid": user,
            "u_trait": self.u_theta(user)
        }
        return obj

    def save_user_profiles(self, user, filepath):
        obj = self.get_user_profiles(user)
        torch.save(obj, filepath)
        print("save user profiles to %s" % os.path.abspath(filepath))

    def get_item_profiles(self, item):
        item = torch.tensor(item, dtype=torch.int64).to(
            self.a.weight.device) if not isinstance(item, torch.Tensor) else item
        obj = {
            "iid": item,
            "ib": self.i_difficulty(item),
            "ia": self.i_discrimination(item),
            "ic": self.i_guess(item),
        }
        return obj

    def save_item_profiles(self, item, filepath):
        obj = self.get_item_profiles(item)
        torch.save(obj, filepath)
        print("save item profiles to %s" % os.path.abspath(filepath))

    def u_theta(self, user):
        return (torch.sigmoid(torch.squeeze(self.theta(user), dim=-1)) - 0.5) * 6

    def i_difficulty(self, item):
        return (torch.sigmoid(torch.squeeze(self.b(item), dim=-1)) - 0.5) * 6
    
    def i_guess(self, item):
        return torch.sigmoid(torch.squeeze(self.c(item), dim=-1))

    def i_discrimination(self, item):
        a = torch.squeeze(self.a(item), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        return a

class EmbedDINANet(_EmbedDINANet):
    def __init__(self, user_num, item_num, know_n, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super().__init__(user_num, item_num, know_n, max_slip, max_guess, *args, **kwargs)

    def forward(self, user, item, knowledge, *args):
        theta = self.u_theta(user)
        slip = self.i_slip(item)
        guess = self.i_guess(item)
        if self.training:
            # 训练
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            # 评估
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            # student i 对 exercise 的潜在作答情况
            return (1 - slip) ** n * guess ** (1 - n)

    def get_user_profiles(self, user):
        user = torch.tensor(user, dtype=torch.int64).to(
            self.theta.weight.device) if not isinstance(user, torch.Tensor) else user
        obj = {
            "uid": user,
            "u_trait": self.u_theta(user)
        }
        return obj

    def save_user_profiles(self, user, filepath):
        obj = self.get_user_profiles(user)
        torch.save(obj, filepath)
        print("save user profiles to %s" % os.path.abspath(filepath))

    def get_item_profiles(self, item):
        item = torch.tensor(item, dtype=torch.int64).to(
            self.guess.weight.device) if not isinstance(item, torch.Tensor) else item
        obj = {
            "iid": item,
            "ia": self.i_guess(item),
            "ib": self.i_slip(item),
        }
        return obj

    def save_item_profiles(self, item, filepath):
        obj = self.get_item_profiles(item)
        torch.save(obj, filepath)
        print("save item profiles to %s" % os.path.abspath(filepath))

    def u_theta(self, user):
        return self.theta(user)

    def i_slip(self, item):
        return torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
    
    def i_guess(self, item):
        return torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)


class EmbedNCDMNet(nn.Module):
    def __init__(self, user_num, item_num, know_n, *args, **kwargs):
        super(EmbedNCDMNet, self).__init__()

        self.knowledge_dim = know_n
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.theta_emb = nn.Embedding(user_num, self.prednet_input_len)
        self.kd_emb = nn.Embedding(item_num, self.prednet_input_len)
        self.ed_emb = nn.Embedding(item_num, self.prednet_input_len)
        self.int_fc = nn.Sequential(
            PosLinear(self.prednet_input_len, self.prednet_len1),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            PosLinear(self.prednet_len1, self.prednet_len2),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            PosLinear(self.prednet_len2, 1),
            nn.Sigmoid()
        )

    def get_user_profiles(self, user):
        user = torch.tensor(user, dtype=torch.int64).to(
            self.theta_emb.weight.device) if not isinstance(user, torch.Tensor) else user
        obj = {
            "uid": user,
            "u_trait": self.u_theta(user)
        }
        return obj

    def save_user_profiles(self, user, filepath):
        obj = self.get_user_profiles(user)
        torch.save(obj, filepath)
        print("save user profiles to %s" % os.path.abspath(filepath))

    def get_item_profiles(self, item):
        item = torch.tensor(item, dtype=torch.int64).to(
            self.kd_emb.weight.device) if not isinstance(item, torch.Tensor) else item
        obj = {
            "iid": item,
            "ib": self.i_difficulty(item),
            "ia": self.i_discrimination(item),
        }
        return obj

    def save_item_profiles(self, item, filepath):
        obj = self.get_item_profiles(item)
        torch.save(obj, filepath)
        print("save item profiles to %s" % os.path.abspath(filepath))

    def save_cdm(self, filepath):
        torch.save(self.int_fc.state_dict(), filepath)
        print("save cdm params to %s" % os.path.abspath(filepath))

    def u_theta(self, user):
        return torch.sigmoid(self.theta_emb(user))

    def i_difficulty(self, item):
        return torch.sigmoid(self.kd_emb(item))

    def i_discrimination(self, item):
        return torch.sigmoid(self.ed_emb(item))

    def forward(self, user, item, i2k):
        # before prednet
        stat_emb = self.u_theta(user)
        k_difficulty = self.i_difficulty(item)
        e_difficulty = self.i_discrimination(item)
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * i2k
        output_1 = self.int_fc(input_x)

        return output_1.view(-1)


def get_net(ctx, cdm, *args, **kwargs):
    if cdm == "ncd":
        net = EmbedNCDMNet(*args, **kwargs)
    elif cdm == "mirt":
        net = EmbedMIRTNet(*args, **kwargs)
    elif cdm == "irt":
        net = EmbedIRTNet(*args, **kwargs)
    elif cdm == "dina":
        net = EmbedDINANet(*args, **kwargs)
    else:
        raise TypeError
    return set_device(net, ctx)


if __name__ == '__main__':
    for name, parameter in EmbedNCDMNet(10, 20, 30).named_parameters():
        print(name)
