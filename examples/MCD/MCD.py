# coding: utf-8
# 2024/1/3 @ Gao Weibo

import logging
from EduCDM import MCD
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from EduData import get_data


# get_data("cdbd-a0910", "../../data")   # Download dataset "cdbd-a0910"

# load data and transform it to the required format
train_data = pd.read_csv("../../data/a0910/train.csv")
valid_data = pd.read_csv("../../data/a0910/valid.csv")
test_data = pd.read_csv("../../data/a0910/test.csv")
df_item = pd.read_csv("../../data/a0910/item.csv")
knowledge_set, item_set = set(), set()
for i, s in df_item.iterrows():
    item_id = s['item_id']
    item_set.add(item_id)
userIds = train_data['user_id'].unique()
meta_data = {'userId': list(userIds), 'itemId': list(item_set)}
train_data = (pd.merge(train_data, df_item, how='left', on='item_id')
              .rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'score': 'response'}))
valid_data = pd.merge(valid_data, df_item, how='left', on='item_id').rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'score': 'response'})
test_data = pd.merge(test_data, df_item, how='left', on='item_id').rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'score': 'response'})

# model training
batch_size = 32
logging.getLogger().setLevel(logging.INFO)
cdm = MCD(meta_data)
cdm.fit(train_data, epoch=1, val_data=valid_data, device="cpu") # cuda

# predict using the trained model
print(cdm.predict(test_data))

# save model
cdm.save("mcd.snapshot")

# load model and evaluate it on the test set
cdm.load("mcd.snapshot")
auc, accuracy = cdm.eval(test_data)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


