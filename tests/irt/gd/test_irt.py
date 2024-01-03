# coding: utf-8
# 2024/1/3 @ Gao Weibo
from EduCDM import GDIRT


def test_train(data, meta, tmp_path):
    df_data = data
    meta_data = meta
    cdm = GDIRT(meta_data)
    cdm.fit(train_data=df_data, epoch=2, val_data=df_data)
    filepath = tmp_path / "irt.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.predict(df_data)
    cdm.eval(df_data)
