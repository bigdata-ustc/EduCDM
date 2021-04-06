# coding: utf-8
# 2021/4/6 @ WangFei
from EduCDM import NCDM


def test_train(data, conf, tmp_path):
    user_num, item_num, knowledge_num = conf
    cdm = NCDM(knowledge_num, item_num, user_num)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "mcd.params"
    cdm.save(filepath)
    cdm.load(filepath)
