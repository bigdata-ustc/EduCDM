# coding: utf-8
# 2023/11/17 @ WangFei
from EduCDM import EMIRT

def test_train(data, meta, tmp_path):
    df_data = data
    meta_data = meta
    cdm = EMIRT(meta_data)
    cdm.fit(train_data=df_data, lr=0.01)
    filepath = tmp_path / "emirt.params"
    # filepath = tmp_path / "emirt.params"
    print(filepath)
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.predict(df_data)
    cdm.eval(df_data)
