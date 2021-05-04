# coding: utf-8
# 2021/5/2 @ liujiayu

from EduCDM import EMIRT


def test_train(data, conf, tmp_path):
    stu_num, prob_num, R, new_data, stu_rec = data
    cdm = EMIRT(R, stu_num, prob_num, dim=1, skip_value=-1)
    cdm.train(lr=1e-3, epoch=30, epsilon=1e-1)
    rmse, mae = cdm.eval([{'user_id': 0, 'item_id': 0, 'score': 1.0}])
    filepath = tmp_path / "irt.params"
    cdm.save(filepath)
    cdm.load(filepath)
    cdm.inc_train(new_data, lr=1e-3, epoch=10)
    cdm.transform(stu_rec)
