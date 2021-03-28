# coding: utf-8
# 2021/3/28 @ liujiayu
from EduCDM import DINA


def test_train(data, conf, tmp_path):
    stu_num, prob_num, know_num, R, q_m = conf
    cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
    cdm.train(epoch=2, epsilon=1e-3)
    rmse, mae = cdm.eval([{'user_id': 0, 'item_id': 0, 'score': 1.0}])
    filepath = tmp_path / "dina.params"
    cdm.save(filepath)
    cdm.load(filepath)
