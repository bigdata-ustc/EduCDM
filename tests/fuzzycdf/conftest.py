# coding: utf-8
# 2021/3/28 @ liujiayu

import random
import numpy as np
import pytest


@pytest.fixture(scope="package")
def conf():
    user_num = 5
    item_num = 2
    know_num = 3
    return user_num, item_num, know_num


@pytest.fixture(scope="package")
def data(conf):
    user_num, item_num, know_num = conf
    q_m = np.zeros(shape=(item_num, know_num))
    for i in range(item_num):
        for j in range(know_num):
            q_m[i, j] = random.randint(0, 1)

    R = -1 * np.ones(shape=(user_num, item_num))
    for i in range(user_num):
        for j in range(item_num):
            R[i, j] = random.randint(-1, 1)
    obj_prob_index = np.array([0])
    sub_prob_index = np.array([1])

    return R, q_m, user_num, item_num, know_num, obj_prob_index, sub_prob_index
