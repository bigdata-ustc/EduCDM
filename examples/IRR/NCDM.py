# coding: utf-8
# 2021/6/19 @ tongshiwei


from EduCDM.IRR import NCDM
import logging
from longling.lib.structure import AttrDict
from longling import set_logging_info
from EduCDM.IRR import pair_etl as etl, point_etl as vt_etl, extract_item

set_logging_info()

params = AttrDict(
    batch_size=256,
    n_neg=10,
    n_imp=10,
    logger=logging.getLogger(),
    hyper_params={"user_num": 4164, "knowledge_num": 123}
)
item_knowledge = extract_item("../../data/a0910/item.csv", params["hyper_params"]["knowledge_num"], params)
train_data, train_df = etl("../../data/a0910/train.csv", item_knowledge, params)
valid_data, _ = vt_etl("../../data/a0910/valid.csv", item_knowledge, params)
test_data, _ = vt_etl("../../data/a0910/test.csv", item_knowledge, params)

cdm = NCDM(
    4163 + 1,
    17746 + 1,
    123,
)
cdm.train(
    train_data,
    valid_data,
    epoch=2,
)
cdm.save("IRR-NCDM.params")

cdm.load("IRR-NCDM.params")
print(cdm.eval(test_data))
