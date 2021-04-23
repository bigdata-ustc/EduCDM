# coding: utf-8
# 2021/3/23 @ tongshiwei

import random
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture(scope="package")
def conf():
    user_num = 5
    item_num = 2
    return user_num, item_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num = conf
    log = []
    for i in range(user_num):
        for j in range(item_num):
            score = random.randint(0, 1)
            log.append((i, j, score))

    user_id, item_id, score = zip(*log)
    batch_size = 4

    dataset = TensorDataset(
        torch.tensor(user_id, dtype=torch.int64),
        torch.tensor(item_id, dtype=torch.int64),
        torch.tensor(score, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size)
