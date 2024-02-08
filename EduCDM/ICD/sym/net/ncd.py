# coding: utf-8
import torch
from torch import nn
from ..pos_linear import PosLinear


class NCDMNet(nn.Module):
    def __init__(self, trait_dim, know_dim):
        super(NCDMNet, self).__init__()

        self.knowledge_dim = know_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.l_dtn_theta_fc = nn.Linear(trait_dim, self.prednet_input_len)
        self.i_dtn_kd_fc = nn.Linear(trait_dim, self.prednet_input_len)
        self.i_dtn_ed_fc = nn.Linear(trait_dim, self.prednet_input_len)
        self.int_fc = nn.Sequential(
            PosLinear(self.prednet_input_len, self.prednet_len1), nn.Sigmoid(),
            nn.Dropout(p=0.5), PosLinear(self.prednet_len1, self.prednet_len2),
            nn.Sigmoid(), nn.Dropout(p=0.5), PosLinear(self.prednet_len2, 1),
            nn.Sigmoid())

    def u_theta(self, u_trait):
        return torch.sigmoid(self.l_dtn_theta_fc(u_trait))

    def i_difficulty(self, v_trait):
        return torch.sigmoid(self.i_dtn_kd_fc(v_trait))

    def i_discrimination(self, v_trait):
        return torch.sigmoid(self.i_dtn_ed_fc(v_trait))

    def forward(self, u_trait, v_trait, v_know):
        theta = self.u_theta(u_trait)

        difficulty = self.i_difficulty(v_trait)
        discrimination = self.i_discrimination(v_trait)

        # prednet
        input_x = discrimination * (theta - difficulty) * v_know
        output_1 = self.int_fc(input_x)

        return output_1.view(-1), theta, discrimination, difficulty

    def int_f(self, theta, a, b, know):
        return self.int_fc(a * (theta - b) * know).view(-1)
