# coding: utf-8
# 2021/3/28 @ liujiayu

import logging
import numpy as np
import pickle
from scipy import stats
from tqdm import tqdm
from collections import namedtuple
from EduCDM import CDM
from .modules import get_LogLikelihood, cal_alpha_mastery, update_A_B, update_theta, update_slip_guess, update_variance

hyper_para = namedtuple("hyperparameters",
                        ["sig_a", "mu_a", "sig_b", "mu_b", "max_s", "min_s", "max_g", "min_g", "mu_theta", "sig_theta"])
default_hyper = hyper_para(1, 0, 1, 0, 0.6, 0, 0.6, 0, 0, 1)


def init_parameters(stu_num, prob_num, know_num, args):  # initialize FuzzyCDF parameters
    a = stats.lognorm.rvs(s=args.sig_a, loc=0, scale=np.exp(args.mu_a), size=(stu_num, know_num))
    b = stats.norm.rvs(loc=args.mu_b, scale=args.sig_b, size=(stu_num, know_num))
    slip = stats.beta.rvs(a=1, b=2, size=prob_num) * (args.max_s - args.min_s) + args.min_s
    guess = stats.beta.rvs(a=1, b=2, size=prob_num) * (args.max_g - args.min_g) + args.min_g
    theta = stats.norm.rvs(loc=args.mu_theta, scale=args.sig_theta, size=stu_num)
    variance = 1 / stats.gamma.rvs(a=4, scale=1 / 6, size=1)
    return a, b, slip, guess, theta, variance


class FuzzyCDF(CDM):
    """
    FuzzyCDF model, training (MCMC) and testing methods
    :param R (array): response matrix, shape = (stu_num, prob_num)
    :param q_m (array): Q matrix, shape = (prob_num, know_num)
    :param stu_num (int): number of students
    :param prob_num (int): number of problems
    :param know_num (int): number of knowledge
    :param obj_prob_index (array): index of all objective problems, shape = (number, )
    :param sub_prob_index (array): index of all subjective problems, shape = (number, )
    :param skip_value (int): skip value in response matrix
    :param args: all hyper-parameters
    """

    def __init__(self, R, q_m, stu_num, prob_num, know_num, obj_prob_index, sub_prob_index, skip_value=-1,
                 args=default_hyper):
        self.args = args
        self.R, self.q_m, self.stu_num, self.prob_num, self.know_num = R, q_m, stu_num, prob_num, know_num
        self.a, self.b, self.slip, self.guess, self.theta, self.variance = init_parameters(stu_num, prob_num, know_num,
                                                                                           self.args)
        self.obj_prob_index, self.sub_prob_index, self.skip_value = obj_prob_index, sub_prob_index, skip_value

    def train(self, epoch, burnin) -> ...:
        A, B, slip, guess = np.copy(self.a), np.copy(self.b), np.copy(self.slip), np.copy(self.guess)
        theta, variance = np.copy(self.theta), np.copy(self.variance)
        estimate_A, estimate_B, estimate_slip, estimate_guess, estimate_theta, estimate_variance = 0, 0, 0, 0, 0, 0
        for iteration in range(epoch):
            update_A_B(A, B, theta, slip, guess, variance, self.R, self.q_m, self.obj_prob_index, self.sub_prob_index,
                       self.skip_value, self.args)
            update_theta(A, B, theta, slip, guess, variance, self.R, self.q_m, self.obj_prob_index, self.sub_prob_index,
                         self.skip_value, self.args)
            update_slip_guess(A, B, theta, slip, guess, variance, self.R, self.q_m, self.obj_prob_index,
                              self.sub_prob_index,
                              self.skip_value, self.args)
            variance = update_variance(A, B, theta, slip, guess, variance, self.R, self.q_m, self.obj_prob_index,
                                       self.sub_prob_index,
                                       self.skip_value)
            if iteration >= burnin:
                estimate_A += A
                estimate_B += B
                estimate_slip += slip
                estimate_guess += guess
                estimate_theta += theta
                estimate_variance += variance
        self.a, self.b, self.slip, self.guess, self.theta, self.variance = estimate_A / (epoch - burnin), estimate_B / (
            epoch - burnin), estimate_slip / (epoch - burnin), estimate_guess / (epoch - burnin), estimate_theta \
            / (epoch - burnin), estimate_variance / (epoch - burnin)

    def eval(self, test_data) -> tuple:
        _, pred_mastery = cal_alpha_mastery(self.a, self.b, self.theta, self.q_m, self.obj_prob_index,
                                            self.sub_prob_index)
        pred_score = (1 - self.slip) * pred_mastery + self.guess * (1 - pred_mastery)
        test_rmse, test_mae = [], []
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            test_rmse.append((pred_score[stu, test_id] - true_score) ** 2)
            test_mae.append(abs(pred_score[stu, test_id] - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"a": self.a, "b": self.b, "theta": self.theta, "slip": self.slip, "guess": self.guess}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.a, self.b, self.theta, self.slip, self.guess = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, epoch, burnin):  # incremental training
        for i in inc_train_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            self.R[stu, test_id] = true_score
        self.train(epoch, burnin)
