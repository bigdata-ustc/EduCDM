# coding: utf-8
# 2021/6/19 @ tongshiwei

import os
import numpy as np
import pandas as pd
from longling import print_time

import torch
from torch.utils.data import TensorDataset, DataLoader


def extract(data_src, params):
    with print_time("loading data from %s" % os.path.abspath(data_src), params.logger):
        df = pd.read_csv(data_src, dtype={"user_id": "int64", "item_id": "int64", "score": "float32"})
        return df


def transform(df, knowledge, *args):
    # 定义数据转换接口
    # raw_data --> batch_data
    dataset = TensorDataset(
        torch.tensor(df["user_id"]),
        torch.tensor(df["item_id"]),
        torch.tensor(np.stack([knowledge[int(item)] for item in df["item_id"]])),
        torch.tensor(df["score"], dtype=torch.float)
    )
    return dataset


def load(transformed_data, params):
    batch_size = params.batch_size

    return DataLoader(transformed_data, batch_size=batch_size)


def etl(filepath, knowledge, params):
    raw_data = extract(filepath, params)
    transformed_data = transform(raw_data, knowledge, params)
    return load(transformed_data, params), raw_data
