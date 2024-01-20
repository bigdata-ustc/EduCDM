# coding: utf-8
# 2023/12/29 @ CSLiJT

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="package")
def meta():
    meta_data = {
        'userId': ['001', '002', '003'],
        'itemId': [0, 1, 2, 3],
        'skill': [0, 1, 2, 3]}
    return meta_data


@pytest.fixture(scope="package")
def data():
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
    return train_data, know_graph
