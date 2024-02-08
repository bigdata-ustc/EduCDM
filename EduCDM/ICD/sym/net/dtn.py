# coding: utf-8
import torch
from torch import nn
from baize.torch.functional import mask_sequence


class DTN(nn.Module):
    def __init__(self, input_dim, know_dim):
        self.know_dim = know_dim
        self.input_dim = input_dim
        self.fea_dim = 64

        super(DTN, self).__init__()
        self.emb = nn.Sequential(nn.Embedding(self.input_dim, self.fea_dim),
                                 # nn.Dropout(p=0.5),
                                 )
        # self.feature_net = nn.Sequential(
        #     # nn.Dropout(p=0.2),
        #     nn.Linear(self.know_dim, self.know_dim),
        #     # nn.Dropout(p=0.5),
        #     # nn.Linear(self.prednet_len2, self.know_dim),
        # )
        # self.atn = nn.MultiheadAttention(self.fea_dim, 4)
        self.feature_net = nn.Sequential(
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(self.fea_dim, self.know_dim))

    def avg_pool(self, data, mask: torch.Tensor):
        # batch_num * emb_dim * max_len => batch_num * emb_dim * 1
        # print(data,mask)
        mask_data = mask_sequence(data, mask)
        rs = torch.sum(mask_data.permute(0, 2, 1), dim=-1)
        len_mask = mask.reshape((-1, 1))
        len_mask = len_mask.expand(len_mask.size()[0], self.know_dim)
        # print(rs.size(),mask.size())
        rs = torch.div(rs, len_mask)
        return rs

    def forward(self, log, mask):
        # emb = mask_sequence(self.emb(log), mask)
        # att_emb = emb.permute(1, 0, 2)
        # att_emb, _ = self.atn(att_emb, att_emb, att_emb)
        # fea = self.feature_net(att_emb)
        # fea = fea.permute(1, 0, 2)

        emb = self.emb(log)
        fea = self.feature_net(emb)

        trait = self.avg_pool(fea, mask)
        return trait
