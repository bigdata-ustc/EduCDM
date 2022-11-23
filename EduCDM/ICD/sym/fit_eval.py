# coding: utf-8

import logging
from torch.utils.data import TensorDataset, DataLoader
import math
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import entropy
from baize.metrics import classification_report, POrderedDict
from baize.torch import fit_wrapper, eval_wrapper
from longling.ML.PytorchHelper import set_device
from EduCDM.ICD.metrics import doa_report, stableness_report
from EduCDM.ICD.etl import dict_etl, Dict2, multi_hot
from .net import ICD, EmbICD


@eval_wrapper
def eval_f(_net, test_data, *args, **kwargs):
    y_true = []
    y_pred = []
    y_label = []

    user_id = []
    item_id = []
    user_theta = []
    item_knowledge = []

    for (
            uid,
            u2i,
            u_mask,
            iid,
            i2u,
            i_mask,
            i2k,
            r,
    ) in tqdm(test_data, "evaluating"):
        pred, theta, *_ = _net(u2i, u_mask, i2u, i_mask, i2k)
        y_pred.extend(pred.tolist())
        y_label.extend([0 if p < 0.5 else 1 for p in pred])
        y_true.extend(r.tolist())

        user_id.extend(uid.tolist())
        item_id.extend(iid.tolist())
        user_theta.extend(theta.tolist())
        item_knowledge.extend(i2k.tolist())

    try:  # pragma: no cover
        if not y_true:
            raise ValueError()
        ret = classification_report(y_true, y_label, y_pred)
    except ValueError:  # pragma: no cover
        ret = POrderedDict()
    ret.update(doa_report(user_id, item_id, item_knowledge, y_true,
                          user_theta))
    return ret


@eval_wrapper
def stableness_eval(net, user, item, u2i, i2u, user_traits, item_traits,
                    batch_size):
    pred_net = net.module if isinstance(net, torch.nn.DataParallel) else net

    new_user_traits = pred_net.get_user_profiles(
        dict_etl(user, u2i, batch_size=batch_size))
    new_item_traits = pred_net.get_item_profiles(
        dict_etl(item, i2u, batch_size=batch_size))

    return stableness_report(
        [user_traits["u_trait"], item_traits['ia'], item_traits['ib']], [
            new_user_traits["u_trait"], new_item_traits['ia'],
            new_item_traits['ib']
        ], ['theta', 'a', 'b'])


@fit_wrapper
def dual_fit_f(_net, batch_data, loss_function, *args, **kwargs):
    (
        _,
        u2i,
        u_mask,
        _,
        i2u,
        i_mask,
        i2k,
        r,
    ) = batch_data
    out, theta, a, b, stat_theta, stat_a, stat_b = _net(
        u2i, u_mask, i2u, i_mask, i2k)

    loss_function["BCE"](out, r)
    loss_function["DTL"](theta, a, b, stat_theta, stat_a, stat_b)
    loss = loss_function["Loss"](out, r, theta, a, b, stat_theta, stat_a,
                                 stat_b)
    return loss


@eval_wrapper
def turning_point(net: (torch.nn.DataParallel, ICD),
                  inc_train_df: pd.DataFrame,
                  dict2: Dict2,
                  inc_dict2: Dict2,
                  i2k,
                  kn,
                  batch_size,
                  ctx,
                  epsilon=1e-2,
                  tolerance=1e-2,
                  logger=logging,
                  *args,
                  **kwargs):
    # return True
    net = net.module if isinstance(net, torch.nn.DataParallel) else net
    valid_inc_users = list(
        set(inc_train_df["user_id"].tolist()) & set(dict2.u2i.keys()))
    valid_inc_items = list(
        set(inc_train_df["item_id"].tolist()) & set(dict2.i2u.keys()))

    if not (valid_inc_items and valid_inc_items):
        return False  # pragma: no cover

    users = []
    items = []
    scores = []
    for user_id in valid_inc_users:
        users.extend([user_id] * len(dict2.u2i[user_id]))
        items.extend(dict2.u2i_i[user_id])
        scores.extend(dict2.u2i_r[user_id])

    for item_id in valid_inc_items:
        items.extend([item_id] * len(dict2.i2u[item_id]))
        users.extend(dict2.i2u_u[item_id])
        scores.extend(dict2.i2u_r[item_id])

    user_traits: dict = net.get_user_profiles(
        dict_etl(set(users), dict2.u2i, batch_size=batch_size))
    item_traits: dict = net.get_item_profiles(
        dict_etl(set(items), dict2.i2u, batch_size=batch_size))

    emb_icd = EmbICD(
        net.cdm.int_f,
        [user_traits["u_trait"], item_traits["ia"], item_traits["ib"]])
    # emb_icd = EmbICD(net.cdm.int_f, net.cdm_name, user_traits, item_traits)
    emb_icd.build_user_id2idx(user_traits["uid"].tolist())
    emb_icd.build_item_id2idx(item_traits["iid"].tolist())

    pred_emb_icd = set_device(emb_icd, ctx)
    know = [multi_hot(i2k[item_id], kn) for item_id in items]

    pred_r = []
    user_grad = []
    item_grad = []
    loss_r = []
    loss_f = set_device(torch.nn.BCELoss(reduction='none'), ctx)
    u_dim = v_dim = None
    data_loader = DataLoader(TensorDataset(
        torch.tensor(emb_icd.user_id2idx(users), dtype=torch.int64),
        torch.tensor(emb_icd.item_id2idx(items), dtype=torch.int64),
        torch.tensor(know, dtype=torch.int64),
        torch.tensor(scores, dtype=torch.float32)),
                             batch_size=1024)
    for uid, iid, ks, rs in tqdm(data_loader, "preparing tp features"):
        pred_emb_icd.zero_grad()
        loss_f.zero_grad()
        _pred_r, theta, a, b = pred_emb_icd(uid, iid, ks)
        if not u_dim:
            u_dim = theta.size()[-1]
            v_dim = a.size()[-1] + b.size()[-1]
        pred_r.append(_pred_r.to("cpu"))
        _pred_r.sum().backward()
        user_grad.append(torch.norm(theta.grad, 2, dim=-1).to("cpu"))
        item_grad.append(
            torch.norm(torch.cat([a.grad, b.grad], dim=-1), 2,
                       dim=-1).to("cpu"))
        loss_r.append(loss_f(_pred_r, rs).to("cpu"))

    df = pd.DataFrame({
        "user_id": users,
        "item_id": items,
        "loss_r": torch.cat(loss_r).tolist(),
        "pred_r": torch.cat(pred_r).tolist(),
        "user_grad": torch.cat(user_grad).tolist(),
        "item_grad": torch.cat(item_grad).tolist(),
    })

    # tp4user
    delta_user = []
    for user in tqdm(valid_inc_users, "TP Learner"):
        r_pos = dict2.u2i_r_dis[user][0]
        r_neg = dict2.u2i_r_dis[user][1]
        m_i = r_pos + r_neg

        delta_r_pos = inc_dict2.u2i_r_dis[user][0]
        delta_r_neg = inc_dict2.u2i_r_dis[user][1]
        n_i = delta_r_pos + delta_r_neg

        delta_r_distribution = [delta_r_pos, delta_r_neg]

        user_df = df[df["user_id"] == user]
        loss_item_r = user_df["loss_r"].sum()
        delta = (user_df["pred_r"] * (1 - user_df["pred_r"])).min()
        beta = user_df["user_grad"].max()

        kl = entropy(delta_r_distribution, [
            sum(user_df["pred_r"]),
            len(user_df["pred_r"]) - sum(user_df["pred_r"])
        ])
        h_delta_r = entropy(delta_r_distribution)

        if not ((delta_r_pos / n_i * r_neg / m_i)**2 +
                (delta_r_neg / n_i * r_pos / m_i)**2) \
                or not beta:  # pragma: no cover
            if entropy(delta_r_distribution, [r_pos, r_neg]) > 0:
                logger.info(
                    "################# user %s inf exceed ##################" %
                    user)
                logger.info(delta_r_pos, delta_r_neg, r_pos, r_neg)
                return True
            else:
                continue

        delta_user.append(
            delta * (kl + h_delta_r - loss_item_r -
                     (m_i + n_i) / n_i * epsilon) /
            (math.sqrt((delta_r_pos / n_i * r_neg / m_i)**2 +
                       (delta_r_neg / n_i * r_pos / m_i)**2) * beta))
    if delta_user and max(delta_user) > tolerance * (u_dim**0.5):
        logger.info("++++++++++++ %s > %s +++++++++++++" %
                    (max(delta_user), tolerance * (u_dim**0.5)))
        return True

    # tp4item
    delta_item = []
    for item in tqdm(valid_inc_items, "TP Item"):
        r_pos = dict2.i2u_r_dis[item][0]
        r_neg = dict2.i2u_r_dis[item][1]
        m_j = r_pos + r_neg

        delta_r_pos = inc_dict2.i2u_r_dis[item][0]
        delta_r_neg = inc_dict2.i2u_r_dis[item][1]
        n_j = delta_r_pos + delta_r_neg

        delta_r_distribution = [delta_r_pos, delta_r_neg]

        item_df = df[df["item_id"] == item]
        loss_item_r = item_df["loss_r"].sum()
        delta = (item_df["pred_r"] * (1 - item_df["pred_r"])).min()
        beta = item_df["item_grad"].max()

        kl = entropy(delta_r_distribution, [
            sum(item_df["pred_r"]),
            len(item_df["pred_r"]) - sum(item_df["pred_r"])
        ])
        h_delta_r = entropy(delta_r_distribution)

        if not (delta_r_pos / n_j * r_neg / m_j +
                delta_r_neg / n_j * r_pos / m_j) or not beta:
            if entropy(delta_r_distribution,
                       [r_pos, r_neg]) > 0:  # pragma: no cover
                logger.info(
                    "################# item %s inf exceed ##################" %
                    item)
                logger.info(delta_r_pos, delta_r_neg, r_pos, r_neg)
                return True
            else:  # pragma: no cover
                continue

        delta_item.append(delta * (kl + h_delta_r - loss_item_r -
                                   (m_j + n_j) / n_j * epsilon) /
                          ((delta_r_pos / n_j * r_neg / m_j +
                            delta_r_neg / n_j * r_pos / m_j) * beta))

    if delta_item and max(delta_item) > tolerance * (v_dim**0.5):
        logger.info("++++++++++++ %s > %s +++++++++++++" %
                    (max(delta_item), tolerance * (v_dim**0.5)))
        return True

    return False
