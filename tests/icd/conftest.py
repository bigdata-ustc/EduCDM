# coding: utf-8
# 2021/4/6 @ WangFei

import random
import pytest
import pandas as pd
from EduCDM.ICD.etl import inc_stream


@pytest.fixture(scope="package")
def conf():
    user_num = 50
    item_num = 20
    knowledge_num = 4
    return user_num, item_num, knowledge_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num, knowledge_num = conf
    i2k = {}
    for i in range(item_num):
        i2k[i] = [random.randint(0, knowledge_num - 1)]
    log = []
    for i in range(user_num):
        for j in range(item_num):
            score = random.randint(0, 1)
            log.append([i, j, score])
    random.shuffle(log)
    df = pd.DataFrame(log, columns=['user_id', 'item_id', 'score'])
    inc_train_df_list = list(inc_stream(df, stream_size=int(len(df) // 50)))

    return inc_train_df_list, i2k
