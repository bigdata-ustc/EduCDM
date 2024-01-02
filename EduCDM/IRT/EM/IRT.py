# coding: utf-8
# 2023/12/26 @ ChenSiHang
import pandas as pd
import sys
import logging
import math 
from typing import Any
import numpy as np
import pickle
from pandas.core.api import DataFrame as DataFrame
from tqdm import tqdm
from scipy import stats
sys.path.append("..")
from ..irt import irt3pl
from EduCDM import CDM, re_index


def init_parameters(prob_num, dim):
    r"""
    Purpose:
        Initialize the parameters of IRT model
    Parameter: 
        prob_num: the number of problems
        dim: the dimension of  student's ability
    Return value:
        alpha: Discrimination of the problems
        beta: Difficulty of the problems
        gamma: Guess parameters of the problems
    """
    alpha = stats.norm.rvs(loc=0.75, scale=0.01, size=(prob_num, dim))
    beta = stats.norm.rvs(size=(prob_num, dim))
    gamma = stats.uniform.rvs(size=prob_num)
    return alpha, beta, gamma


def init_prior_prof_distribution(dim):
    r"""
    Purpose:
        Initialize the prior distribution of student abilities
    Parameter:
        dim: Dimension of student's ability
    Return value:
        prof: Uniform distribution from -4 to 4, 100*dim matrix
        norm_dis: Normalized distribution of student abilities
    """
    prof = stats.uniform.rvs(loc=-4, scale=8, size=(100, dim))  # shape = (100,dim)
    dis = stats.multivariate_normal.pdf(prof, mean=np.zeros(dim), cov=np.identity(dim))
    norm_dis = dis / np.sum(dis)  # shape = (100,)
    return prof, norm_dis


def get_Likelihood(a, b, c, prof, R):
    r"""
    Purpose:
        get the likelihood function
    Parameter:
        a: Discrimination of the problems
        b: Difficulty of the problems
        c: Guess parameters of the problems
        prof: Normalized distribution of student's ability
        R: matrix of item response
    Return value:
        prof_prob: Probability matrix for 100 ability levels of people answering each question correctly
        prob_stu: Probability matrix of which ability level a student belongs to
    """
    stu_num, prob_num = R.shape[0], R.shape[1]
    prof_prob = irt3pl(np.sum(a * (np.expand_dims(prof, axis=1) - b), axis=-1), 1, 0, c)  # shape = (100, prob_num)
    tmp1, tmp2 = np.zeros(shape=(prob_num, stu_num)), np.zeros(shape=(prob_num, stu_num))
    tmp1[np.where(R == 1)[1], np.where(R == 1)[0]] = 1 # shape = (prob_num, stu_num)
    tmp2[np.where(R == 0)[1], np.where(R == 0)[0]] = 1 # shape = (prob_num, stu_num)
    prob_stu = np.exp(np.dot(np.log(prof_prob + 1e-9), tmp1) + np.dot(np.log(1 - prof_prob + 1e-9), tmp2))
    return prof_prob, prob_stu # shape = (100, prob_num), (100, stu_num)


def update_prior(prior_dis, prof_stu_like):
    r"""
    Purpose:
        update the prior distribution of student abilities
    Parameter:
        prior_dis: prior distribution of student abilities
        prof_stu_like: Probability matrix of which ability level a student belongs to
    Return value:
        update_prior_dis: updated prior distribution of student abilities
        norm_dis_like: Normalized distribution of student abilities
    """
    dis_like = prof_stu_like * np.expand_dims(prior_dis, axis=1)  # shape = (100, stu_num)
    norm_dis_like = dis_like / np.sum(dis_like, axis=0) # shape = (100, stu_num)
    update_prior_dis = np.sum(norm_dis_like, axis=1) / np.sum(norm_dis_like) # shape = (100,)
    return update_prior_dis, norm_dis_like # shape = (100,), (100, stu_num)


def update_irt(a, b, c, D, prof, R, r_ek, s_ek, lr, epoch=10, epsilon=1e-3):
    r"""
    Purpose:
        update the parameters of IRT model
    Parameter:
        a: Discrimination of the problems
        b: Difficulty of the problems
        c: Guess parameters of the problems
        D: the value of D
        prof: Normalized distribution of student's ability
        R: matrix of item response
        r_ek: the number of students who answered correctly
        s_ek: the number of students who answered
        lr: learning rate
        epoch: the number of iterations
        epsilon: threshold of convergence
    Return value:
        a: Discrimination of the problems
        b: Difficulty of the problems
        c: Guess parameters of the problems
    """
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
    r"""
    IRT model, training (EM) and testing methods
    Args:
        meta_data: a dictionary containing all the userIds, itemIds, and skills.
        dim: int
            dimension of student/problem embedding, MIRT for dim > 1
        skip_value: int
            Unavailable value in matrix need to be skipped
    """
    def __init__(self, meta_data: dict, dim=1, skip_value=-1):
        super(IRT, self).__init__()
        self.id_reindex, _ = re_index(meta_data)
        self.stu_num = len(self.id_reindex['userId'])
        self.prob_num = len(self.id_reindex['itemId'])
        self.dim = dim
        self.skip_value = skip_value
        self.a, self.b, self.c = init_parameters(self.prob_num, dim)
        self.D = 1.702
        self.prof, self.prior_dis = init_prior_prof_distribution(dim)
        self.stu_prof = np.zeros(shape=(self.stu_num, dim))
        
        
    def transform__(self, train_data: pd.DataFrame):
        r"""
        transform the train data to matrix of item response
        
        Args:
            train_data: training dataset
        Return value:
            R: matrix of item response
        """
        R = np.full((self.stu_num, self.prob_num), self.skip_value)
        for index, i in tqdm(train_data.iterrows(), "transforming"):
            stu, test_id, true_score = i['userId'], i['itemId'], i['response']
            re_stu_id, re_item_id = self.id_reindex['userId'][stu], self.id_reindex['itemId'][test_id]
            R[re_stu_id, re_item_id] = true_score
        return R


    def fit(self, train_data: pd.DataFrame, lr, epoch=10, epoch_m=10, epsilon=1e-3):
        r"""
        Train the IRT model
        This function uses the EM algorithm to train the model, in which the M step uses gradient descent to maximize parameters.
        
        Args:
            train_data: training dataset
            test_data: test dataset
            stu_num: the number of students
            prob_num: the number of problems
        Return value:
            R: matrix of item response
            test_set: test dataset
        """
        R = self.transform__(train_data)
        a, b, c = np.copy(self.a), np.copy(self.b), np.copy(self.c)
        prior_dis = np.copy(self.prior_dis)
        for iteration in range(epoch):
            a_tmp, b_tmp, c_tmp, prior_dis_tmp = np.copy(a), np.copy(b), np.copy(c), np.copy(prior_dis)
            prof_prob_like, prof_stu_like = get_Likelihood(a, b, c, self.prof,  R)
            prior_dis, norm_dis_like = update_prior(prior_dis, prof_stu_like)
            r_1 = np.zeros(shape=(self.stu_num, self.prob_num))
            r_1[np.where( R == 1)[0], np.where( R == 1)[1]] = 1
            r_ek = np.dot(norm_dis_like, r_1)  # shape = (100, prob_num)
            r_1[np.where( R != self.skip_value)[0], np.where( R != self.skip_value)[1]] = 1
            s_ek = np.dot(norm_dis_like, r_1)  # shape = (100, prob_num)
            a, b, c = update_irt(a, b, c, self.D, self.prof,  R, r_ek, s_ek, lr, epoch_m, epsilon)
            change = max(np.max(np.abs(a - a_tmp)), np.max(np.abs(b - b_tmp)), np.max(np.abs(c - c_tmp)),
                         np.max(np.abs(prior_dis_tmp - prior_dis_tmp)))
            if iteration > 20 and change < epsilon:
                break
        self.a, self.b, self.c, self.prior_dis = a, b, c, prior_dis
        self.stu_prof = self.Get_stu_ability(R)


    def predict_proba(self, val_data) -> pd.DataFrame:
        r"""
        calculate the probability
        
        Args:
            None
        Return value:
            pred_score: the probability of students response correctly
        """
        # pred_score_1 = irt3pl(np.sum(self.a * (np.expand_dims(self.stu_prof, axis=1) - self.b), axis=-1), 1, 0, self.c)
        userIds, itemIds, responses = [], [], []
        for index, i in tqdm(val_data.iterrows(), "predicting"):
            stu, test_id = i['userId'], i['itemId']
            re_stu_id, re_item_id = self.id_reindex['userId'][stu], self.id_reindex['itemId'][test_id]
            userIds.append(re_stu_id)
            itemIds.append(re_item_id)
            responses.append(irt3pl(np.sum(self.a[re_item_id] * (np.expand_dims(self.stu_prof[re_stu_id], axis=1) - self.b[re_item_id]), axis=-1), 1, 0, self.c[re_item_id]))
        return pd.DataFrame({'userId': userIds, 'itemId': itemIds, 'response': responses})
        
        
    def predict(self, val_data) -> pd.DataFrame:
        r"""
        predict the probability
        
        Args:
            None
        Return value:
            df_pred: the probability
        """
        irt_proba = self.predict_proba(val_data)
        irt_proba.loc[irt_proba['response'] < 0.5, 'response'] = 0
        irt_proba.loc[irt_proba['response'] >= 0.5, 'response'] = 1
        df_pred = pd.DataFrame(irt_proba)
        return df_pred
        

    def eval(self, val_data: pd.DataFrame):
        r"""
        Evaluate the IRT model
        
        Args:
            val_data: validation dataset
        Return value:
            test_rmse: RMSE of test dataset
            test_mae: MAE of test dataset
        """
        pred_score = self.predict_proba(val_data)
        # print(pred_score)
        test_rmse, test_mae = [], []
        for index, i in tqdm(val_data.iterrows(), "evaluating"):
            stu, test_id, true_score = i['userId'], i['itemId'], i['response']
            re_stu_id, re_item_id = self.id_reindex['userId'][stu], self.id_reindex['itemId'][test_id]
            test_rmse.append((pred_score.iloc[re_stu_id, re_item_id] - true_score) ** 2)
            test_mae.append(abs(pred_score.iloc[re_stu_id, re_item_id] - true_score))
        return np.sqrt(np.average(test_rmse)), np.average(test_mae)


    def save(self, filepath: str):
        r"""
        save the parameters of IRT model
        
        Args:
            filepath: the path of file
        Return value:
            None
        """
        with open(filepath, 'wb') as file:
            pickle.dump({"a": self.a, "b": self.b, "c": self.c, "prof": self.stu_prof}, file)
            logging.info("save parameters to %s" % filepath)


    def load(self, filepath: str):
        r"""
        load the parameters of IRT model
        
        Args:
            filepath: the path of file
        Return value:
            None
        """
        with open(filepath, 'rb') as file:
            self.a, self.b, self.c, self.stu_prof = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)


    def Get_stu_ability(self, records, lr=1e-3, epoch=10, epsilon=1e-3):  
        r"""
        Transform the records to student's ability by MLE method 
        can evaluate multiple students' states simultaneously, thus output shape = (stu_num, dim)
        initialization stu_prof, shape = (stu_num, dim)
        
        Args:
            records: records of students' answers
            lr: learning rate
            epoch: the number of iterations
            epsilon: threshold of convergence
        Return value:
            stu_prof: student's ability distribution
        """
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