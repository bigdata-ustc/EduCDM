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
    knowledge_num = 3
    return user_num, item_num, knowledge_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num, knowledge_num = conf
    log = []
    for i in range(user_num):
        for j in range(item_num):
            k = [0] * knowledge_num
            k[random.randint(0, knowledge_num - 1)] = 1
            score = random.randint(0, 1)
            log.append((i, j, k, score))

    user_id, item_id, knowledge, score = zip(*log)
    batch_size = 4

    dataset = TensorDataset(
        torch.tensor(user_id, dtype=torch.int64),
        torch.tensor(item_id, dtype=torch.int64),
        torch.tensor(knowledge, dtype=torch.float),
        torch.tensor(score, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size)
