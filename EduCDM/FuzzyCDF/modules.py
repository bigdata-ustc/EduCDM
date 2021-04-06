# coding: utf-8
# 2021/3/28 @ liujiayu
# Modules in FuzzyCDF

import numpy as np
from scipy import stats


def cal_alpha_mastery(A, B, theta, q_m, obj_prob_index, sub_prob_index):  # calculate proficiency on knows and probs
    stu_num, prob_num = len(theta), q_m.shape[0]
    alpha = 1 / (1 + np.exp(-1.7 * A * (theta.reshape([-1, 1]) - B)))
    mastery = np.zeros((stu_num, prob_num))
    for i in range(stu_num):
        stu_i = alpha[i] * q_m  # shape = (prob_num, know_num)
        if len(obj_prob_index) > 0:
            mastery[i][obj_prob_index] = np.min((stu_i + 2 * (1 - q_m))[obj_prob_index], axis=1)
        if len(sub_prob_index) > 0:
            mastery[i][sub_prob_index] = np.max(stu_i[sub_prob_index], axis=1)
    return alpha, mastery


def get_LogLikelihood(A, B, theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index, skip_value=-1):
    # calculate log-likelihood for each response log
    _, mastery = cal_alpha_mastery(A, B, theta, q_m, obj_prob_index, sub_prob_index)
    stu_num, prob_num = R.shape[0], R.shape[1]
    x = (1 - slip) * mastery + guess * (1 - mastery)
    result = np.zeros((stu_num, prob_num))
    if len(obj_prob_index) > 0:
        result[:, obj_prob_index] = (np.log(x + 1e-9) * R + np.log(1 - x + 1e-9) * (1 - R))[:, obj_prob_index]
    if len(sub_prob_index) > 0:
        result[:, sub_prob_index] = np.log(stats.norm.pdf(R, loc=x, scale=variance))[:, sub_prob_index]

    result[np.where(R == skip_value)[0], np.where(R == skip_value)[1]] = 0  # skip logs
    return result  # shape = (stu_num, prob_num)


# ---below are updating processes in MCMC for FuzzyCDF---
def update_A_B(A, B, theta, slip, guess, variance, R, q_m, obj_prob_index, sub_prob_index, skip_value, args):
    know_num = A.shape[1]
    new_A = A + 0.3 * stats.norm.rvs(size=A.shape)
    new_B = B + 0.3 * stats.norm.rvs(size=B.shape)
    for know in range(know_num):
        tempA = np.copy(A)
        tempB = np.copy(B)
        tempA[:, know] = np.copy(new_A[:, know])
        tempB[:, know] = np.copy(new_B[:, know])

        l_0 = get_LogLikelihood(A, B, theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index, skip_value)
        l_1 = get_LogLikelihood(tempA, tempB, theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index,
                                skip_value)

        log_p0 = np.sum(l_0, axis=1) + np.log(stats.norm.pdf(x=B[:, know], loc=args.mu_b, scale=args.sig_b) + 1e-9) + \
            np.log(stats.lognorm.pdf(x=A[:, know], loc=0, scale=np.exp(args.mu_a), s=args.sig_a) + 1e-9)
        log_p1 = np.sum(l_1, axis=1) + np.log(stats.norm.pdf(x=tempB[:, know], loc=args.mu_b, scale=args.sig_b) + 1e-9)\
            + np.log(stats.lognorm.pdf(x=tempA[:, know], loc=0, scale=np.exp(args.mu_a), s=args.sig_a) + 1e-9)
        accept_prob = np.exp(np.minimum(log_p1 - log_p0, 0))  # avoid overflow in exp
        mask = accept_prob >= np.random.random(1)
        A[mask, know] = new_A[mask, know]
        B[mask, know] = new_B[mask, know]


def update_theta(A, B, theta, slip, guess, variance, R, q_m, obj_prob_index, sub_prob_index, skip_value, args):
    new_theta = theta + 0.1 * stats.norm.rvs(size=theta.shape)

    l_0 = get_LogLikelihood(A, B, theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index, skip_value)
    l_1 = get_LogLikelihood(A, B, new_theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index, skip_value)

    log_p0 = np.sum(l_0, axis=1) + np.log(stats.norm.pdf(x=theta, loc=args.mu_theta, scale=args.sig_theta) + 1e-9)
    log_p1 = np.sum(l_1, axis=1) + np.log(stats.norm.pdf(x=new_theta, loc=args.mu_theta, scale=args.sig_theta) + 1e-9)
    accept_prob = np.exp(np.minimum(log_p1 - log_p0, 0))  # avoid overflow in exp
    mask = accept_prob >= np.random.random(1)
    theta[mask] = new_theta[mask]


def update_slip_guess(A, B, theta, slip, guess, variance, R, q_m, obj_prob_index, sub_prob_index, skip_value, args):
    new_slip = np.abs(slip + 0.2 * stats.norm.rvs(size=slip.shape) - 0.1)
    new_guess = np.abs(guess + 0.2 * stats.norm.rvs(size=guess.shape) - 0.1)

    l_0 = get_LogLikelihood(A, B, theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index, skip_value)
    l_1 = get_LogLikelihood(A, B, theta, R, q_m, new_slip, new_guess, variance, obj_prob_index, sub_prob_index,
                            skip_value)

    log_p0 = np.sum(l_0, axis=0) + np.log(stats.beta.pdf(x=slip / (args.max_s - args.min_s), a=1, b=2) + 1e-9) + np.log(
        stats.beta.pdf(x=guess / (args.max_g - args.min_g), a=1, b=2) + 1e-9)
    log_p1 = np.sum(l_1, axis=0) + np.log(stats.beta.pdf(x=new_slip / (args.max_s - args.min_s), a=1, b=2) + 1e-9) + \
        np.log(stats.beta.pdf(x=new_guess / (args.max_g - args.min_g), a=1, b=2) + 1e-9)
    accept_prob = np.exp(np.minimum(log_p1 - log_p0, 0))  # avoid overflow in exp
    mask = accept_prob >= np.random.random(1)
    slip[mask] = new_slip[mask]
    guess[mask] = new_guess[mask]


def update_variance(A, B, theta, slip, guess, variance, R, q_m, obj_prob_index, sub_prob_index, skip_value):
    new_var = np.maximum(variance - 0.01 + 0.02 * stats.norm.rvs(size=variance.shape), 0)

    l_0 = get_LogLikelihood(A, B, theta, R, q_m, slip, guess, variance, obj_prob_index, sub_prob_index, skip_value)
    l_1 = get_LogLikelihood(A, B, theta, R, q_m, slip, guess, new_var, obj_prob_index, sub_prob_index, skip_value)

    l_0[:, obj_prob_index] = 0
    l_1[:, obj_prob_index] = 0

    log_p0 = np.sum(l_0) + np.log(stats.gamma.pdf(x=1 / (variance + 1e-9), a=4, scale=1 / 6) + 1e-9)
    log_p1 = np.sum(l_1) + np.log(stats.gamma.pdf(x=1 / (new_var + 1e-9), a=4, scale=1 / 6) + 1e-9)
    accept_prob = np.exp(np.minimum(log_p1 - log_p0, 0))  # avoid overflow in exp
    if accept_prob >= np.random.random(1):
        variance = new_var
    return variance
