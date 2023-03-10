# coding: utf-8
# 2023/3/8 @ WangFei
from EduCDM import KaNCD


def test_train(data, conf, tmp_path):
    user_num, item_num, knowledge_num = conf
    cdm = KaNCD(exer_n=item_num, student_n=user_num, knowledge_n=knowledge_num, mf_type='mf', dim=2)
    cdm.train(data, data, epoch_n=2)
    cdm = KaNCD(exer_n=item_num, student_n=user_num, knowledge_n=knowledge_num, mf_type='gmf', dim=2)
    cdm.train(data, data, epoch_n=2)
    cdm = KaNCD(exer_n=item_num, student_n=user_num, knowledge_n=knowledge_num, mf_type='ncf1', dim=2)
    cdm.train(data, data, epoch_n=2)
    cdm = KaNCD(exer_n=item_num, student_n=user_num, knowledge_n=knowledge_num, mf_type='ncf2', dim=2)
    cdm.train(data, data, epoch_n=2)
    filepath = tmp_path / "kancd.params"
    cdm.save(filepath)
    cdm.load(filepath)
