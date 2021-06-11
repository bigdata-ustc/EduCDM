# coding: utf-8
# 2021/5/2 @ liujiayu

import logging
import numpy as np
import pickle
from tqdm import tqdm
from scipy import stats
from ..irt import irt3pl
from EduCDM import CDM


def init_parameters(prob_num, dim):
    alpha = stats.norm.rvs(loc=0.75, scale=0.01, size=(prob_num, dim))
    beta = stats.norm.rvs(size=(prob_num, dim))
    gamma = stats.uniform.rvs(size=prob_num)
    return alpha, beta, gamma


def init_prior_prof_distribution(dim):
    prof = stats.uniform.rvs(loc=-4, scale=8, size=(100, dim))  # shape = (100,dim)
    dis = stats.multivariate_normal.pdf(prof, mean=np.zeros(dim), cov=np.identity(dim))
    norm_dis = dis / np.sum(dis)  # shape = (100,)
    return prof, norm_dis


def get_Likelihood(a, b, c, prof, R):
    stu_num, prob_num = R.shape[0], R.shape[1]
    prof_prob = irt3pl(np.sum(a * (np.expand_dims(prof, axis=1) - b), axis=-1), 1, 0, c)  # shape = (100, prob_num)
    tmp1, tmp2 = np.zeros(shape=(prob_num, stu_num)), np.zeros(shape=(prob_num, stu_num))
    tmp1[np.where(R == 1)[1], np.where(R == 1)[0]] = 1
    tmp2[np.where(R == 0)[1], np.where(R == 0)[0]] = 1
    prob_stu = np.exp(np.dot(np.log(prof_prob + 1e-9), tmp1) + np.dot(np.log(1 - prof_prob + 1e-9), tmp2))
    return prof_prob, prob_stu


def update_prior(prior_dis, prof_stu_like):
    dis_like = prof_stu_like * np.expand_dims(prior_dis, axis=1)
    norm_dis_like = dis_like / np.sum(dis_like, axis=0)
    update_prior_dis = np.sum(norm_dis_like, axis=1) / np.sum(norm_dis_like)
    return update_prior_dis, norm_dis_like


def update_irt(a, b, c, D, prof, R, r_ek, s_ek, lr, epoch=10, epsilon=1e-3):
    for iteration in range(epoch):
        a_tmp, b_tmp, c_tmp = np.copy(a), np.copy(b), np.copy(c)
        prof_prob, _ = get_Likelihood(a, b, c, prof, R)
        common_term = (r_ek - s_ek * prof_prob) / prof_prob / (1 - c + 1e-9)  # shape = (100, prob_num)
        a_1 = np.transpose(
            D * common_term * (prof_prob - c) * np.transpose(np.expand_dims(prof, axis=1) - b, (2, 0, 1)), (1, 2, 0))
        b_1 = D * common_term * (c - prof_prob)
        a_grad = np.sum(a_1, axis=0)
        b_grad = a * np.expand_dims(np.sum(b_1, axis=0), axis=1)
        c_grad = np.sum(common_term, axis=0)
        a = a + lr * a_grad
        b = b + lr * b_grad
        c = np.clip(c + lr * c_grad, 0, 1)
        change = max(np.max(np.abs(a - a_tmp)), np.max(np.abs(b - b_tmp)), np.max(np.abs(c - c_tmp)))
        if iteration > 5 and change < epsilon:
            break
    return a, b, c


class IRT(CDM):
    """
    IRT model, training (EM) and testing methods
    Parameters
    ----------
    R: numpy.array
        response matrix, shape = (stu_num, prob_num)
    stu_num: int
        number of students
    prob_num: int
        number of problems
    dim: int
        dimension of student/problem embedding, MIRT for dim > 1
    skip_value: int
        skip value in response matrix
    ----------
    """
    def __init__(self, R, stu_num, prob_num, dim=1, skip_value=-1):
        super(IRT, self).__init__()
        self.R, self.skip_value = R, skip_value
        self.stu_num, self.prob_num, self.dim = stu_num, prob_num, dim
        self.a, self.b, self.c = init_parameters(prob_num, dim)  # IRT parameters
        self.D = 1.702
        self.prof, self.prior_dis = init_prior_prof_distribution(dim)
        self.stu_prof = np.zeros(shape=(stu_num, dim))

    def train(self, lr, epoch, epoch_m=10, epsilon=1e-3):
        a, b, c = np.copy(self.a), np.copy(self.b), np.copy(self.c)
        prior_dis = np.copy(self.prior_dis)
        for iteration in range(epoch):
            a_tmp, b_tmp, c_tmp, prior_dis_tmp = np.copy(a), np.copy(b), np.copy(c), np.copy(prior_dis)
            prof_prob_like, prof_stu_like = get_Likelihood(a, b, c, self.prof, self.R)
            prior_dis, norm_dis_like = update_prior(prior_dis, prof_stu_like)

            r_1 = np.zeros(shape=(self.stu_num, self.prob_num))
            r_1[np.where(self.R == 1)[0], np.where(self.R == 1)[1]] = 1
            r_ek = np.dot(norm_dis_like, r_1)  # shape = (100, prob_num)
            r_1[np.where(self.R != self.skip_value)[0], np.where(self.R != self.skip_value)[1]] = 1
            s_ek = np.dot(norm_dis_like, r_1)  # shape = (100, prob_num)
            a, b, c = update_irt(a, b, c, self.D, self.prof, self.R, r_ek, s_ek, lr, epoch_m, epsilon)
            change = max(np.max(np.abs(a - a_tmp)), np.max(np.abs(b - b_tmp)), np.max(np.abs(c - c_tmp)),
                         np.max(np.abs(prior_dis_tmp - prior_dis_tmp)))
            if iteration > 20 and change < epsilon:
                break
        self.a, self.b, self.c, self.prior_dis = a, b, c, prior_dis
        self.stu_prof = self.transform(self.R)

    def eval(self, test_data) -> tuple:
        pred_score = irt3pl(np.sum(self.a * (np.expand_dims(self.stu_prof, axis=1) - self.b), axis=-1), 1, 0, self.c)
        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            test_rmse.append((pred_score[stu, test_id] - true_score) ** 2)
            test_mae.append(abs(pred_score[stu, test_id] - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"a": self.a, "b": self.b, "c": self.c, "prof": self.stu_prof}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.a, self.b, self.c, self.stu_prof = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, lr=1e-3, epoch=10, epsilon=1e-3):  # incremental training
        for i in inc_train_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            self.R[stu, test_id] = true_score
        self.train(lr, epoch, epsilon=epsilon)

    def transform(self, records, lr=1e-3, epoch=10, epsilon=1e-3):  # MLE for evaluating students' state
        # can evaluate multiple students' states simultaneously, thus output shape = (stu_num, dim)
        # initialization stu_prof, shape = (stu_num, dim)
        if len(records.shape) == 1:  # one student
            records = np.expand_dims(records, axis=0)
        _, prof_stu_like = get_Likelihood(self.a, self.b, self.c, self.prof, records)
        stu_prof = self.prof[np.argmax(prof_stu_like, axis=0)]

        for iteration in range(epoch):
            prof_tmp = np.copy(stu_prof)
            ans_prob = irt3pl(np.sum(self.a * (np.expand_dims(stu_prof, axis=1) - self.b), axis=-1), 1, 0, self.c)
            ans_1 = self.D * (records - ans_prob) / ans_prob * (ans_prob - self.c) / (1 - self.c + 1e-9)
            ans_1[np.where(records == self.skip_value)[0], np.where(records == self.skip_value)[1]] = 0
            prof_grad = np.dot(ans_1, self.a)
            stu_prof = stu_prof - lr * prof_grad
            change = np.max(np.abs(stu_prof - prof_tmp))
            if iteration > 5 and change < epsilon:
                break
        return stu_prof  # shape = (stu_num, dim)
