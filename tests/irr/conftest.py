# coding: utf-8
# 2021/3/23 @ tongshiwei

import pandas as pd
import random
import pytest
from longling.lib.structure import AttrDict
from EduCDM.IRR import pair_etl, point_etl, extract_item
import logging
from copy import deepcopy


@pytest.fixture(scope="package")
def conf():
    user_num = 5
    item_num = 2
    knowledge_num = 3
    return user_num, item_num, knowledge_num


@pytest.fixture(scope="package")
def params(conf):
    user_num, item_num, knowledge_num = conf
    return AttrDict(
        logger=logging,
        user_num=user_num,
        item_num=item_num,
        knowledge_num=knowledge_num,
        n_neg=1,
        n_imp=1,
        hyper_params={"user_num": user_num},
        batch_size=4
    )


@pytest.fixture(scope="package")
def source(tmpdir_factory, conf):
    user_num, item_num, knowledge_num = conf

    d = tmpdir_factory.mktemp("irr")
    log_path = d / "log.csv"
    item_path = d / "item.csv"

    knowledge = []
    for j in range(item_num):
        knowledge.append([j, [random.randint(1, knowledge_num)]])

    pd.DataFrame(knowledge, columns=["item_id", "knowledge_code"]).to_csv(item_path)

    log = []
    for i in range(user_num):
        for j in range(item_num):
            score = random.randint(0, 1)
            log.append((i, j, score))

    pd.DataFrame(log, columns=["user_id", "item_id", "score"]).to_csv(log_path)

    return log_path, item_path


@pytest.fixture(scope="package")
def knowledge(source, params):
    _, item_path = source
    return extract_item(item_path, params.knowledge_num, params)


@pytest.fixture(scope="package")
def train_data(source, knowledge, params):
    log_path, _ = source
    data, _ = pair_etl(log_path, knowledge, params)
    return data


@pytest.fixture(scope="package")
def zero_train_data(source, knowledge, params):
    log_path, _ = source
    params_0 = dict(params.items())
    params_0["n_neg"] = 0
    params_0["n_imp"] = 0
    params_0 = AttrDict(**params_0)
    data, _ = pair_etl(log_path, knowledge, params_0)
    return data


@pytest.fixture(scope="package")
def test_data(source, knowledge, params):
    log_path, _ = source
    data, _ = point_etl(log_path, knowledge, params)
    return data
