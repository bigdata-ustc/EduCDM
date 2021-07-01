# coding: utf-8
# 2021/6/19 @ tongshiwei

from EduCDM.IRR import MIRT


def test_irr_irt(train_data, test_data, params, tmp_path):
    cdm = MIRT(params.user_num, params.item_num, params.knowledge_num)
    cdm.train(train_data, test_data=test_data, epoch=2)
    filepath = tmp_path / "irr.params"
    cdm.save(filepath)
    cdm.load(filepath)


def test_irt(zero_train_data, test_data, params, tmp_path):
    cdm = MIRT(params.user_num, params.item_num, params.knowledge_num, zeta=0)
    cdm.train(zero_train_data, test_data=test_data, epoch=2)
    filepath = tmp_path / "irr.params"
    cdm.save(filepath)
    cdm.load(filepath)
