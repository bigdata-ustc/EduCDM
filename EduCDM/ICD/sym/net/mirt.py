# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from .dtn import DTN
from EduCDM.MIRT.MIRT import irt2pl


class MIRTNet(nn.Module):
    def __init__(self, trait_dim, a_range=0.1, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.l_dtn_theta = nn.Linear(trait_dim, trait_dim)
        self.i_dtn_a = nn.Linear(trait_dim, trait_dim)
        self.i_dtn_b = nn.Linear(trait_dim, 1)
        self.a_range = a_range

    def forward(self, u_trait, v_trait, *args):
        theta = self.u_theta(u_trait)
        b = self.i_difficulty(v_trait)
        a = self.i_discrimination(v_trait)

        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')

        return self.irf(theta, a, b, **self.irf_kwargs), theta, a, b

    @classmethod
    def int_f(cls, theta, a, b, *args, **kwargs):
        return irt2pl(theta, a, b, F=torch)

    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)

    def u_theta(self, u_trait):
        return (torch.sigmoid(torch.squeeze(self.l_dtn_theta(u_trait), dim=-1)) - 0.5) * 6

    def i_difficulty(self, v_trait):
        return (torch.sigmoid(torch.squeeze(self.i_dtn_b(v_trait), dim=-1)) - 0.5) * 6

    def i_discrimination(self, v_trait):
        a = torch.squeeze(self.i_dtn_a(v_trait), dim=-1)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:  # pragma: no cover
            a = F.softplus(a)
        return a
