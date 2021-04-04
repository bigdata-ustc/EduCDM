# coding: utf-8
# 2021/3/28 @ liujiayu
import logging
import numpy as np
import json
from EduCDM import FuzzyCDF


# type of problems
obj_prob_index = np.loadtxt("../../data/math2015/Math1/obj_prob_index.csv", delimiter=',', dtype=int)
sub_prob_index = np.loadtxt("../../data/math2015/Math1/sub_prob_index.csv", delimiter=',', dtype=int)
# Q matrix
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

cdm = FuzzyCDF(R, q_m, stu_num, prob_num, know_num, obj_prob_index, sub_prob_index, skip_value=-1)

cdm.train(epoch=10, burnin=5)
cdm.save("fuzzycdf.params")

cdm.load("fuzzycdf.params")
rmse, mae = cdm.eval(test_set)
print("RMSE, MAE are %.6f, %.6f" % (rmse, mae))

# ---incremental training
new_data = [{'user_id': 0, 'item_id': 2, 'score': 0.0}, {'user_id': 1, 'item_id': 1, 'score': 1.0}]
cdm.inc_train(new_data, epoch=10, burnin=5)
