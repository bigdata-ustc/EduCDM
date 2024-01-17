# coding: utf-8
# 2024/01/15 @ CSLiJT
import logging
from EduCDM import HierCDF
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np


train_data = pd.DataFrame({
    'userId': [
        '001', '001', '001', '001', '002', '002',
        '002', '002', '003', '003', '003', '003'],
    'itemId': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    'response': [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
})
know_graph = pd.DataFrame({
    'source': [0, 0, 1],
    'target': [1, 2, 3]
})
q_matrix = np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1]
])

train_data['skill'] = 0
for id in range(train_data.shape[0]):
    item_id = train_data.loc[id, 'itemId']
    concepts = np.where(
        q_matrix[item_id] > 0)[0].tolist()
    train_data.loc[id, 'skill'] = str(concepts)
meta_data = {'userId': [], 'itemId': [], 'skill': []}
meta_data['userId'] = train_data['userId'].unique().tolist()
meta_data['itemId'] = train_data['itemId'].unique().tolist()
meta_data['skill'] = [i for i in range(q_matrix.shape[1])]

hiercdm = HierCDF(meta_data, know_graph, hidd_dim=32)
hiercdm.fit(
    train_data,
    val_data=train_data,
    batch_size=1, epoch=3, lr=0.01)
hiercdm.save('./hiercdf.pt')
new_hiercdm = HierCDF(meta_data, know_graph, hidd_dim=32)
new_hiercdm.load('./hiercdf.pt')
new_hiercdm.fit(
    train_data,
    val_data=train_data,
    batch_size=1, epoch=1, lr=0.01)
new_hiercdm.eval(train_data)