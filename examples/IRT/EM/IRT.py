# coding: utf-8
# 2021/5/2 @ liujiayu
import logging
import numpy as np
import pandas as pd
from EduCDM import EMIRT

train_data = pd.read_csv("../../../data/a0910/train.csv")
valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("../../../data/a0910/test.csv")

stu_num = max(max(train_data['user_id']), max(test_data['user_id']))
prob_num = max(max(train_data['item_id']), max(test_data['item_id']))

R = -1 * np.ones(shape=(stu_num, prob_num))
R[train_data['user_id']-1, train_data['item_id']-1] = train_data['score']

test_set = []
for i in range(len(test_data)):
    row = test_data.iloc[i]
    test_set.append({'user_id':int(row['user_id'])-1, 'item_id':int(row['item_id'])-1, 'score':row['score']})

logging.getLogger().setLevel(logging.INFO)

cdm = EMIRT(R, stu_num, prob_num, dim=1, skip_value=-1)  # IRT, dim > 1 is MIRT

cdm.train(lr=1e-3, epoch=2)
cdm.save("irt.params")

cdm.load("irt.params")
rmse, mae = cdm.eval(test_set)
print("RMSE, MAE are %.6f, %.6f" % (rmse, mae))

# ---incremental training
new_data = [{'user_id': 0, 'item_id': 2, 'score': 0.0}, {'user_id': 1, 'item_id': 1, 'score': 1.0}]
cdm.inc_train(new_data, lr=1e-3, epoch=2)

# ---evaluate user's state
stu_rec = np.random.randint(-1, 2, size=prob_num)
dia_state = cdm.transform(stu_rec)
print("user's state is " + str(dia_state))
