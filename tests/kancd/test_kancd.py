# coding: utf-8
# 2023/11/17 @ WangFei
from EduCDM import KaNCD


def test_train(data, meta, tmp_path):
    df_data = data
    meta_data = meta
    cdm = KaNCD(meta_data, mf_type='mf')
    cdm.fit(train_data=df_data, epoch=1, val_data=df_data)
    cdm = KaNCD(meta_data, mf_type='gmf')
    cdm.fit(train_data=df_data, epoch=1, val_data=df_data)
    cdm = KaNCD(meta_data, mf_type='ncf1')
    cdm.fit(train_data=df_data, epoch=1, val_data=df_data)
    cdm = KaNCD(meta_data, mf_type='ncf2')
    cdm.fit(train_data=df_data, epoch=1, val_data=df_data)
    filepath = tmp_path / "kancd.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.predict(df_data)
    cdm.eval(df_data)
