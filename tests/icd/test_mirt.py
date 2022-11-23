# coding: utf-8
# 2021/4/23 @ tongshiwei

from EduCDM.ICD.ICD import ICD
# from EduCDM import ICD


def test_train(data, conf, tmp_path):
    user_n, item_n, know_n = conf
    cdm = ICD('mirt', user_n, item_n, know_n)
    log, i2k = data
    cdm.train(log, i2k)
    cdm.save()
    cdm.load()


def test_exception(data, conf, tmp_path):
    try:
        user_n, item_n, know_n = conf
        cdm = ICD('mirt', user_n, item_n, know_n)
        log, i2k = data
        cdm.train(log, i2k)
        cdm.save()
        cdm.load()
    except ValueError:
        print(ValueError)
