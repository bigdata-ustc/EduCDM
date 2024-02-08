# coding: utf-8
# 2022/2/1 @ tongshiwei
import pandas as pd
from longling.ML.metrics import POrderedDict
import numpy as np
from tqdm import tqdm


def doa_report(user, item, know, score, theta):
    df = pd.DataFrame({
        "user_id": user,
        "item_id": item,
        "score": score,
        "theta": theta,
        "knowledge": know
    })
    ground_truth = []

    for _, group_df in tqdm(df.groupby("item_id"), "formatting item df"):
        ground_truth.append(group_df["score"].values)
        ground_truth.append(1 - group_df["score"].values)

    knowledges = []
    knowledge_item = []
    knowledge_user = []
    knowledge_truth = []
    knowledge_theta = []
    for user, item, score, theta, knowledge in tqdm(
            df[["user_id", "item_id", "score", "theta", "knowledge"]].values,
            "formatting knowledge df"):
        if isinstance(theta, list):
            for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta_i)
        else:  # pragma: no cover
            for i, knowledge_i in enumerate(knowledge):
                if knowledge_i == 1:
                    knowledges.append(i)
                    knowledge_item.append(item)
                    knowledge_user.append(user)
                    knowledge_truth.append(score)
                    knowledge_theta.append(theta)

    knowledge_df = pd.DataFrame({
        "knowledge": knowledges,
        "user_id": knowledge_user,
        "item_id": knowledge_item,
        "score": knowledge_truth,
        "theta": knowledge_theta
    })
    knowledge_ground_truth = []
    knowledge_prediction = []
    for _, group_df in knowledge_df.groupby("knowledge"):
        _knowledge_ground_truth = []
        _knowledge_prediction = []
        for _, item_group_df in group_df.groupby("item_id"):
            _knowledge_ground_truth.append(item_group_df["score"].values)
            _knowledge_prediction.append(item_group_df["theta"].values)
        knowledge_ground_truth.append(_knowledge_ground_truth)
        knowledge_prediction.append(_knowledge_prediction)

    return POrderedDict(doa_eval(knowledge_ground_truth, knowledge_prediction))


def doa_eval(y_true, y_pred):
    """
    >>> import numpy as np
    >>> y_true = [
    ...     [np.array([1, 0, 1])],
    ...     [np.array([0, 1, 1])]
    ... ]
    >>> y_pred = [
    ...     [np.array([.5, .4, .6])],
    ...     [np.array([.2, .3, .5])]
    ... ]
    >>> doa_eval(y_true, y_pred)['doa']
    1.0
    >>> y_pred = [
    ...     [np.array([.4, .5, .6])],
    ...     [np.array([.3, .2, .5])]
    ... ]
    >>> doa_eval(y_true, y_pred)['doa']
    0.5
    """
    doa = []
    doa_support = 0
    z_support = 0
    for knowledge_label, knowledge_pred in tqdm(zip(y_true, y_pred),
                                                "doa metrics"):
        _doa = 0
        _z = 0
        for label, pred in zip(knowledge_label, knowledge_pred):
            if sum(label) == len(label) or sum(label) == 0:
                continue
            pos_idx = []
            neg_idx = []
            for i, _label in enumerate(label):
                if _label == 1:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)
            pos_pred = pred[pos_idx]
            neg_pred = pred[neg_idx]
            invalid = 0
            for _pos_pred in pos_pred:
                _doa += len(neg_pred[neg_pred < _pos_pred])
                invalid += len(neg_pred[neg_pred == _pos_pred])
            _z += (len(pos_pred) * len(neg_pred)) - invalid
        if _z > 0:
            doa.append(_doa / _z)
            z_support += _z
            doa_support += 1
    return {
        "doa": np.mean(doa),
        "doa_know_support": doa_support,
        "doa_z_support": z_support,
    }


def stableness_report(traits: list, new_traits: list, keys: list):
    ret = {}
    a_dim = None
    b_dim = None
    for trait, new_trait, key in zip(traits, new_traits, keys):
        if key == "b" and b_dim is None:
            b_dim = trait.size()[-1] if len(trait.size()) > 1 else 1
        if key == "a" and a_dim is None:
            a_dim = trait.size()[-1]

        ret[key] = {}
        delta = (trait - new_trait).abs()
        ret[key]['delta'] = delta.sum().item()
        ret[key]['delta_ave'] = delta.mean().item()
        ret[key]['support'] = len(trait)

    ret["user"] = ret["theta"]
    ret["item"] = {
        "delta":
        ret["a"]["delta"] + ret["b"]["delta"],
        "delta_ave":
        (ret["a"]["delta_ave"] * a_dim + ret["b"]["delta_ave"] * b_dim) /
        (a_dim + b_dim),
        "support":
        ret["a"]["support"],
    }
    macro = ret["user"]["delta_ave"] + ret["item"]["delta_ave"]
    micro = ret["user"]["support"] * ret["user"]["delta_ave"] + ret["item"][
        "support"] * ret["item"]["delta_ave"]
    ret["macro_ave"] = macro / 2
    ret["micro_ave"] = micro / (ret["user"]["support"] +
                                ret["item"]["support"])
    return POrderedDict(ret)
