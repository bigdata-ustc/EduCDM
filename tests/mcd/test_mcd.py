# coding: utf-8
# 2021/3/23 @ tongshiwei
from EduCDM import MCD


def test_train(data, conf, tmp_path):
    user_num, item_num = conf
    cdm = MCD(user_num, item_num, 10)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "mcd.params"
    cdm.save(filepath)
    cdm.load(filepath)
