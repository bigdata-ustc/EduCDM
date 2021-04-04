# coding: utf-8
# 2021/3/28 @ liujiayu
from EduCDM import FuzzyCDF


def test_train(data, tmp_path):
    stu_num, prob_num, know_num, R, q_m, obj_prob_index, sub_prob_index, new_data = data
    cdm = FuzzyCDF(R, q_m, stu_num, prob_num, know_num, obj_prob_index, sub_prob_index, skip_value=-1)
    cdm.train(epoch=10, burnin=5)
    rmse, mae = cdm.eval([{'user_id': 0, 'item_id': 0, 'score': 1.0}])
    filepath = tmp_path / "fuzzycdf.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.inc_train(new_data, epoch=10, burnin=5)
