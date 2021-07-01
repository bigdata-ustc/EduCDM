# coding: utf-8
# 2021/6/19 @ tongshiwei

from EduCDM.IRR import DINA


def test_irr_dina(train_data, test_data, params, tmp_path):
    cdm = DINA(params.user_num, params.item_num, params.knowledge_num)
    cdm.train(train_data, test_data=test_data, epoch=2)
    filepath = tmp_path / "irr.params"
    cdm.save(filepath)
    cdm.load(filepath)


def test_irt(zero_train_data, test_data, params, tmp_path):
    cdm = DINA(params.user_num, params.item_num, params.knowledge_num, zeta=0)
    cdm.train(zero_train_data, test_data=test_data, epoch=2)
    filepath = tmp_path / "irr.params"
    cdm.save(filepath)
    cdm.load(filepath)
