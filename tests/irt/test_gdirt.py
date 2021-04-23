# coding: utf-8
# 2021/4/23 @ tongshiwei

from EduCDM import GDIRT


def test_train(data, conf, tmp_path):
    user_num, item_num = conf
    cdm = GDIRT(user_num, item_num)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "mcd.params"
    cdm.save(filepath)
    cdm.load(filepath)
