# coding: utf-8

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from ICD.etl import item2knowledge

__all__ = ["etl", "item2knowledge", "extract", "transform"]


def transform(logs_df, i2k, kn, batch_size, user_set=None, item_set=None):
    user = []
    item = []
    response = []
    for u, i, r in zip(logs_df['user_id'].tolist(), logs_df['item_id'].tolist(), logs_df['score'].tolist()):
        if (user_set and u not in user_set) or (item_set and i not in item_set):
            continue
        user.append(u)
        item.append(i)
        response.append(r)
    if not response:
        return None
    knows = torch.zeros((len(item), kn))
    
    for idx, iid in enumerate(item):
        knows[idx][i2k[iid]] = 1.0

    dataset = TensorDataset(
        torch.tensor(user, dtype=torch.int64),
        torch.tensor(item, dtype=torch.int64),
        knows,
        torch.tensor(response, dtype=torch.float32)
    )

    return DataLoader(dataset, batch_size=batch_size)


def extract(filepath):
    df = pd.read_csv(filepath)
    return df


def etl(filepath, i2k, kn, batch_size):
    logs_df = extract(filepath)
    return transform(logs_df, i2k, kn, batch_size)
