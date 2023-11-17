# coding: utf-8
# 2023/11/17 @ WangFei

import pytest
import pandas as pd
import random


@pytest.fixture(scope="package")
def meta():
    meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}
    return meta_data


@pytest.fixture(scope="package")
def data(conf):
    meta_data = meta
    item_skills = []
    skll_n = len(meta_data['skill'])
    for itemid in meta_data['itemId']:
        item_skills.append(meta['skill'][random.randint(0, skll_n - 1)])
    userIds, itemIds, skills, responses = []
    for user in meta_data['userId']:
        for i, item in enumerate(meta_data['itemId']):
            userIds.append(user)
            itemIds.append(item)
            skills.append(item_skills[i])
            responses.append(random.randint(0, 1))

    df_data = pd.DataFrame({'userId': userIds, 'itemId': itemIds, 'skill': skills, 'response': responses})
    return df_data
