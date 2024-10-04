# coding: utf-8
# 2022/4/23 @ tongshiwei

from EduCDM.ICD.ICD import ICD
from baize import config_logging


def test_train(data, conf, tmp_path):
    cdm = 'ncd'
    epoch = 1
    weight_decay = 0.1
    inner_metrics = True
    logger = config_logging(logger="ICD", console_log_level="info")
    alpha = 0.9
    ctx = 'cpu'
    beta = 0.95
    warmup_ratio = 0.1
    tolerance = 0.001
    max_u2i = None
    max_i2u = None
    hyper_tag = False
    wfs = None

    train_data, stream_num, df_item, meta_data = data
    cdm = ICD(cdm, meta_data, epoch, weight_decay, inner_metrics, logger, alpha, ctx)
    cdm.fit(train_data, df_item, stream_num, beta, warmup_ratio, tolerance, max_u2i, max_i2u, hyper_tag, wfs)
    
    filepath = tmp_path / "kancd.params"
    cdm.save(filepath)
    cdm.load(filepath)
    print(cdm.predict(train_data))


def test_exception(data, conf, tmp_path):
    try:
        cdm = 'ncd'
        epoch = 1
        weight_decay = 0.1
        inner_metrics = True
        logger = config_logging(logger="ICD", console_log_level="info")
        alpha = 0.9
        ctx = 'cpu'
        beta = 0.95
        warmup_ratio = 0.1
        tolerance = 0.001
        max_u2i = None
        max_i2u = None
        hyper_tag = False
        wfs = None
        train_data, stream_num, df_item, meta_data = data
        cdm = ICD(cdm, meta_data, epoch, weight_decay, inner_metrics, logger, alpha, ctx)
        cdm.fit(train_data, df_item, stream_num, beta, warmup_ratio, tolerance, max_u2i, max_i2u, hyper_tag, wfs)

        filepath = tmp_path / "kancd.params"
        cdm.save(filepath)
        cdm.load(filepath)
        print(cdm.predict(train_data))

    except ValueError:
        print(ValueError)
