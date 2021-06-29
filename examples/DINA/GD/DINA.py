# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from EduCDM import GDDINA
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

train_data = pd.read_csv("../../../data/a0910/train.csv")
valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("../../../data/a0910/test.csv")
item_data = pd.read_csv("../../../data/a0910/item.csv")

knowledge_num = 123


def code2vector(x):
    vector = [0] * knowledge_num
    for k in eval(x):
        vector[k - 1] = 1
    return vector


item_data["knowledge"] = item_data["knowledge_code"].apply(code2vector)
item_data.drop(columns=["knowledge_code"], inplace=True)

train_data = pd.merge(train_data, item_data, on="item_id")
valid_data = pd.merge(valid_data, item_data, on="item_id")
test_data = pd.merge(test_data, item_data, on="item_id")

batch_size = 32


def transform(x, y, z, k, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(k, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], data["knowledge"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = GDDINA(4164, 17747, knowledge_num)

cdm.train(train, valid, epoch=2)
cdm.save("dina.params")

cdm.load("dina.params")
auc, accuracy = cdm.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
