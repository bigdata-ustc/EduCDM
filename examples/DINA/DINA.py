# coding: utf-8
# 2021/3/28 @ liujiayu
import logging
import numpy as np
import json
from EduCDM import DINA

q_m = np.loadtxt("../../data/math2015/Math1/q_m.csv", dtype=int, delimiter=',')
prob_num, know_num = q_m.shape[0], q_m.shape[1]

# training data
with open("../../data/math2015/Math1/train_data.json", encoding='utf-8') as file:
    train_set = json.load(file)
stu_num = max([x['user_id'] for x in train_set]) + 1
R = -1 * np.ones(shape=(stu_num, prob_num))
for log in train_set:
    R[log['user_id'], log['item_id']] = log['score']

# testing data
with open("../../data/math2015/Math1/test_data.json", encoding='utf-8') as file:
    test_set = json.load(file)

logging.getLogger().setLevel(logging.INFO)

cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)

cdm.train(epoch=2, epsilon=1e-3)
cdm.save("dina.params")

cdm.load("dina.params")
rmse, mae = cdm.eval(test_set)
print("RMSE: %.6f, MAE: %.6f" % (rmse, mae))

# ---evaluate user's state
stu_rec = np.array([0, 1, -1, 0, -1, 0, 1, 1, 0, 1, 0, 1, 0, -1, -1, -1, -1, 0, 1, -1])
dia_id, dia_state = cdm.transform(stu_rec)
print("id of user's state is %d, state is " + str(dia_state) % dia_id)
