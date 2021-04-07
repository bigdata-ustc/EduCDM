# coding: utf-8
# 2021/4/6 @ WangFei

import random
import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture(scope="package")
def conf():
    user_num = 5
    item_num = 2
    knowledge_num = 4
    return user_num, item_num, knowledge_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num, knowledge_num = conf
    knowledge_embs = np.zeros((item_num, knowledge_num))
    for i in range(item_num):
        for j in range(knowledge_num):
            knowledge_embs[i][j] = random.randint(0, 1)
    log = []
    for i in range(user_num):
        for j in range(item_num):
            score = random.randint(0, 1)
            log.append((i, j, knowledge_embs[j], score))

    user_id, item_id, knowledge_emb, score = zip(*log)
    batch_size = 4

    dataset = TensorDataset(
        torch.tensor(user_id, dtype=torch.int64),
        torch.tensor(item_id, dtype=torch.int64),
        torch.tensor(knowledge_emb, dtype=torch.int64),
        torch.tensor(score, dtype=torch.float)
    )
    return DataLoader(dataset, batch_size=batch_size)
