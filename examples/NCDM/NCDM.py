# coding: utf-8
# 2021/4/1 @ WangFei
import logging
from EduCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from EduData import get_data


# get_data("cdbd-a0910", "../../data")
train_data = pd.read_csv("../../data/a0910/train.csv")
valid_data = pd.read_csv("../../data/a0910/valid.csv")
test_data = pd.read_csv("../../data/a0910/test.csv")
df_item = pd.read_csv("../../data/a0910/item.csv")
knowledge_set, item_set = set(), set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    knowledge_set.update(knowledge_codes)
    item_set.add(item_id)
userIds = train_data['user_id'].unique()
meta_data = {'userId': userIds, 'itemId': list(item_set), 'skill': list(knowledge_set)}
train_data = (pd.merge(train_data, df_item, how='left', on='item_id')
              .rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'knowledge_code': 'skill', 'score': 'response'}))
valid_data = pd.merge(valid_data, df_item, how='left', on='item_id').rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'knowledge_code': 'skill', 'score': 'response'})
test_data = pd.merge(test_data, df_item, how='left', on='item_id').rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'knowledge_code': 'skill', 'score': 'response'})

batch_size = 32

logging.getLogger().setLevel(logging.INFO)
cdm = NCDM(meta_data)
cdm.fit(train_data, epoch=1, val_data=valid_data, device="cuda")
print(cdm.predict(test_data))
cdm.save("ncdm.snapshot")

cdm.load("ncdm.snapshot")
auc, accuracy = cdm.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


