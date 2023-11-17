# coding: utf-8
# 2023/11/17 @ WangFei
from EduCDM import NCDM


def test_train(data, meta, tmp_path):
    df_data = data
    meta_data = meta
    cdm = NCDM(meta_data)
    cdm.fit(train_data=df_data, epoch=2, val_data=df_data)
    filepath = tmp_path / "mcd.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.eval(df_data)
