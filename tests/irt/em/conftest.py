# coding: utf-8
# 2021/5/2 @ liujiayu

import random
import numpy as np
import pytest


@pytest.fixture(scope="package")
def conf():
    user_num = 5
    item_num = 2
    return user_num, item_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num = conf

    R = -1 * np.ones(shape=(user_num, item_num))
    for i in range(user_num):
        for j in range(item_num):
            R[i, j] = random.randint(-1, 1)

    new_data = [{'user_id': 1, 'item_id': 1, 'score': 1.0}]

    stu_rec = np.ones(item_num)
    for i in range(item_num):
        stu_rec[i] = random.randint(-1, 1)

    return user_num, item_num, R, new_data, stu_rec
