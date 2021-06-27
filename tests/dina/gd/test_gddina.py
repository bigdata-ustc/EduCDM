# coding: utf-8
# 2021/4/23 @ tongshiwei

import pytest
from EduCDM import GDDINA


@pytest.mark.parametrize("ste", [True, False])
def test_train(data, conf, tmp_path, ste):
    user_num, item_num, knowledge_num = conf
    cdm = GDDINA(user_num, item_num, knowledge_num, ste=ste)
    cdm.train(data, test_data=data, epoch=2)
    filepath = tmp_path / "dina.params"
    cdm.save(filepath)
    cdm.load(filepath)
