import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 

'''
interaction functions of CDM
'''

def irt2pl(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return 1 / (1 + torch.exp(-1.7*item_offset*(user_emb - item_emb) ))

def mirt2pl(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return 1 / (1 + torch.exp(- torch.sum(torch.mul(user_emb, item_emb), axis=1).reshape(-1,1) + item_offset))

def sigmoid_dot(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return torch.sigmoid(torch.sum(torch.mul(user_emb, item_emb), axis = -1)).reshape(-1,1)

def dot(user_emb: torch.Tensor, item_emb: torch.Tensor, item_offset: torch.Tensor):
    return torch.sum(torch.mul(user_emb, item_emb), axis = -1).reshape(-1,1)

itf_dict = {
    'irt': irt2pl,
    'mirt': mirt2pl,
    'mf': dot, 
    'sigmoid-mf': sigmoid_dot
}