# coding: utf-8
# 2021/3/28 @ liujiayu

import logging
import numpy as np
from tqdm import tqdm
import pickle
from EduCDM import CDM


def initial_all_knowledge_state(know_num):
    state_num = 2 ** know_num
    all_states = np.zeros((state_num, know_num))
    for i in range(state_num):
        k, quotient, residue = 1, i // 2, i % 2
        while True:
            all_states[i, know_num - k] = residue
            if quotient <= 0:
                break
            quotient, residue = quotient // 2, quotient % 2
            k += 1
    return all_states


def init_parameters(stu_num, prob_num):
    slip = np.zeros(shape=prob_num) + 0.2
    guess = np.zeros(shape=prob_num) + 0.2
    theta = np.zeros(shape=stu_num)  # index of state
    return theta, slip, guess


class DINA(CDM):
    """
        DINA model, training (EM) and testing methods
        :param R (array): response matrix, shape = (stu_num, prob_num)
        :param q_m (array): Q matrix, shape = (prob_num, know_num)
        :param stu_num (int): number of students
        :param prob_num (int): number of problems
        :param know_num (int): number of knowledge
        :param skip_value (int): skip value in response matrix
    """

    def __init__(self, R, q_m, stu_num, prob_num, know_num, skip_value=-1):
        self.R, self.q_m, self.state_num, self.skip_value = R, q_m, 2 ** know_num, skip_value
        self.stu_num, self.prob_num, self.know_num = stu_num, prob_num, know_num
        self.theta, self.slip, self.guess = init_parameters(stu_num, prob_num)
        self.all_states = initial_all_knowledge_state(know_num)  # shape = (state_num, know_num)
        state_prob = np.transpose(np.sum(q_m, axis=1, keepdims=True) - np.dot(q_m, np.transpose(self.all_states)))
        self.eta = 1 - (state_prob > 0)  # state covers knowledge of problem (1: yes), shape = (state_num, prob_num)

    def train(self, epoch, epsilon) -> ...:
        like = np.zeros(shape=(self.stu_num, self.state_num))  # likelihood
        post = np.zeros(shape=(self.stu_num, self.state_num))  # posterior
        theta, slip, guess, tmp_R = np.copy(self.theta), np.copy(self.slip), np.copy(self.guess), np.copy(self.R)
        tmp_R[np.where(self.R == self.skip_value)[0], np.where(self.R == self.skip_value)[1]] = 0
        for iteration in range(epoch):
            post_tmp, slip_tmp, guess_tmp = np.copy(post), np.copy(slip), np.copy(guess)
            answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)
            for s in range(self.state_num):
                log_like = np.log(answer_right[s, :] + 1e-9) * self.R + np.log(1 - answer_right[s, :] + 1e-9) * (
                    1 - self.R)
                log_like[np.where(self.R == self.skip_value)[0], np.where(self.R == self.skip_value)[1]] = 0
                like[:, s] = np.exp(np.sum(log_like, axis=1))
            post = like / np.sum(like, axis=1, keepdims=True)
            i_l = np.expand_dims(np.sum(post, axis=0), axis=1)  # shape = (state_num, 1)
            r_jl = np.dot(np.transpose(post), tmp_R)  # shape = (state_num, prob_num)
            r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)
            i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0)
            guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1

            change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)),
                         np.max(np.abs(guess - guess_tmp)))
            theta = np.argmax(post, axis=1)
            if iteration > 20 and change < epsilon:
                break
        self.theta, self.slip, self.guess = theta, slip, guess

    def eval(self, test_data) -> tuple:
        pred_score = (1 - self.slip) * self.eta + self.guess * (1 - self.eta)
        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            test_rmse.append((pred_score[self.theta[stu], test_id] - true_score) ** 2)
            test_mae.append(abs(pred_score[self.theta[stu], test_id] - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"theta": self.theta, "slip": self.slip, "guess": self.guess}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.theta, self.slip, self.guess = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, epoch, epsilon):  # incremental training
        for i in inc_train_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            self.R[stu, test_id] = true_score
        self.train(epoch, epsilon)

    def transform(self, records):  # MLE for evaluating student's state
        # max_like_id: diagnose which state among all_states the student belongs to
        # dia_state: binaray vector of length know_num, 0/1 indicates whether masters the knowledge
        answer_right = (1 - self.slip) * self.eta + self.guess * (1 - self.eta)
        log_like = records * np.log(answer_right + 1e-9) + (1 - records) * np.log(1 - answer_right + 1e-9)
        log_like[:, np.where(records == self.skip_value)[0]] = 0
        max_like_id = np.argmax(np.exp(np.sum(log_like, axis=1)))
        dia_state = self.all_states[max_like_id]
        return max_like_id, dia_state
