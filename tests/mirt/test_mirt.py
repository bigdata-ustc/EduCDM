# coding: utf-8
# 2021/4/23 @ tongshiwei

from EduCDM import MIRT
import pytest


def test_train(data, conf, tmp_path):
    user_num, item_num = conf
    cdm = MIRT(user_num, item_num, 10)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "mcd.params"
    cdm.save(filepath)
    cdm.load(filepath)


def test_exception(data, conf, tmp_path):
    try:
        user_num, item_num = conf
        cdm = MIRT(user_num, item_num, 10, a_range=100)
        cdm.train(data, test_data=data, epoch=2)
        filepath = tmp_path / "mcd.params"
        cdm.save(filepath)
        cdm.load(filepath)
    except ValueError:
        print(ValueError)
