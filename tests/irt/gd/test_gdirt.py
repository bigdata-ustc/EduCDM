# coding: utf-8
# 2021/4/23 @ tongshiwei

from EduCDM import GDIRT
import pytest


def test_train(data, conf, tmp_path):
    user_num, item_num = conf
    cdm = GDIRT(user_num, item_num)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "mcd.params"
    cdm.save(filepath)
    cdm.load(filepath)


def test_raises(data, conf, tmp_path):
    with pytest.raises(ValueError) as exec_info:
        user_num, item_num = conf
        cdm = GDIRT(user_num, item_num, value_range=10, a_range=100)
        cdm.train(data, test_data=data, epoch=2)
        filepath = tmp_path / "mcd.params"
        cdm.save(filepath)
        cdm.load(filepath)
    assert exec_info.type == ValueError
    assert exec_info.value.args[0] == "ValueError:theta,a,b may contains nan!  The value_range or a_range is too large."
