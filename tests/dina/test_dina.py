# coding: utf-8
# 2023/12/29 @ CSLiJT
from EduCDM import DINA


def test_train(data, meta, tmp_path):
    df_data = data
    meta_data = meta
    cdm = DINA(meta_data, max_slip=0.5, max_guess=0.5)
    cdm.fit(train_data=df_data, epoch=2, val_data=df_data)
    filepath = tmp_path / "dina.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.predict(df_data)
    cdm.eval(df_data)
