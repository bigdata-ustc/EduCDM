# coding: utf-8
# 2023/12/29 @ CSLiJT
from EduCDM import HierCDF


def test_train(data, meta, tmp_path):
    df_data, know_graph = data
    meta_data = meta
    cdm = HierCDF(meta_data, know_graph, hidd_dim=32)
    cdm.fit(train_data=df_data, epoch=2, val_data=df_data)
    filepath = tmp_path / "hiercdf.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.predict(df_data)
    cdm.eval(df_data)
