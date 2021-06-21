# coding: utf-8
# 2021/4/23 @ tongshiwei

from EduCDM import GDDINA


def test_train(data, conf, tmp_path):
    user_num, item_num, knowledge_num = conf
    cdm = GDDINA(user_num, item_num, knowledge_num)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dina.params"
    cdm.save(filepath)
    cdm.load(filepath)
