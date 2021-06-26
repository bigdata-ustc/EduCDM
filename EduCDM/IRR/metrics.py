# coding: utf-8
# 2021/6/19 @ tongshiwei

from longling import as_list
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict
from longling.lib.formatter import table_format, series_format


def result_format(data: dict, col=None):
    table = OrderedDict()
    series = OrderedDict()
    for key, value in data.items():
        if isinstance(value, dict):
            table[key] = value
        else:
            series[key] = value

    _ret = []
    if table:
        _ret.append(table_format(table))
    if series:
        _ret.append(series_format(series, col=col))

    return "\n".join(_ret)


def ranking_auc(ranked_label):
    """
    Examples
    --------
    >>> ranking_auc([1, 1, 0, 0, 0])
    1.0
    >>> ranking_auc([0, 1, 0, 1, 0])
    0.5
    >>> ranking_auc([1, 0, 1, 0, 0])
    0.8333333333333334
    >>> ranking_auc([0, 0, 0, 1, 1])
    0.0
    """
    pos_num = sum(ranked_label)
    neg_num = len(ranked_label) - sum(ranked_label)
    if pos_num * neg_num == 0:  # pragma: no cover
        return 1
    return sum(
        [len(ranked_label[i + 1:]) - sum(ranked_label[i + 1:]) for i, score in enumerate(ranked_label) if score == 1]
    ) / (pos_num * neg_num)


def ranking_report(y_true, y_pred, k: (int, list) = None, coerce="ignore", pad_pred=-100, bottom=True):
    import numpy as np
    from collections import OrderedDict
    from sklearn.metrics import (
        label_ranking_average_precision_score,
        ndcg_score, label_ranking_loss, coverage_error
    )
    assert coerce in {"ignore", "abandon", "raise", "padding"}
    k = as_list(k) if k is not None else [1, 3, 5, 10]
    results = {
        "auc": [],
        "map": [],
        "mrr": [],
        "coverage_error": [],
        "ranking_loss": [],
        "len": [],
        "support": [],
    }
    if bottom:
        results.update({
            "map(B)": [],
            "mrr(B)": [],
        })
    k_results = {}
    for _k in k:
        k_results[_k] = {
            "ndcg@k": [],
            "precision@k": [],
            "recall@k": [],
            "f1@k": [],
            "len@k": [],
            "support@k": [],
        }
        if bottom:
            k_results[_k].update({
                "ndcg@k(B)": [],
                "precision@k(B)": [],
                "recall@k(B)": [],
                "f1@k(B)": [],
                "len@k(B)": [],
                "support@k(B)": [],
            })
    suffix = [""]
    if bottom:
        suffix += ["(B)"]

    for label, pred in tqdm(zip(y_true, y_pred), "ranking metrics"):
        results["map"].append(label_ranking_average_precision_score([label], [pred]))
        if bottom:
            results["map(B)"].append(label_ranking_average_precision_score(
                [(1 - np.asarray(label)).tolist()],
                [(-np.asarray(pred)).tolist()]
            ))

        try:
            results["coverage_error"].append(coverage_error([label], [pred]))
        except ValueError:  # pragma: no cover
            pass
        try:
            results["ranking_loss"].append(label_ranking_loss([label], [pred]))
        except ValueError:  # pragma: no cover
            pass
        results["len"].append(len(label))
        results["support"].append(1)
        label_pred = list(sorted(zip(label, pred), key=lambda x: x[1], reverse=True))
        sorted_label = list(zip(*label_pred))[0]
        results["auc"].append(ranking_auc(sorted_label))
        try:
            results["mrr"].append(1 / (np.asarray(sorted_label).nonzero()[0][0] + 1))
        except IndexError:  # pragma: no cover
            pass
        try:
            if bottom:
                results["mrr(B)"].append(1 / (np.asarray(sorted_label[::-1]).nonzero()[0][0] + 1))
        except IndexError:  # pragma: no cover
            pass
        for _k in k:
            for _suffix in suffix:
                if _suffix == "":
                    _label_pred = deepcopy(label_pred)
                    if len(_label_pred) < _k:
                        if coerce == "abandon":  # pragma: no cover
                            continue
                        elif coerce == "raise":  # pragma: no cover
                            raise ValueError("Not enough value: %s vs target %s" % (len(_label_pred), _k))
                        elif coerce == "padding":
                            _label_pred += [(0, pad_pred)] * (_k - len(_label_pred))
                    k_label_pred = label_pred[:_k]
                    total_label = sum(label)
                else:
                    inv_label_pred = [(1 - _l, -p) for _l, p in label_pred][::-1]
                    if len(inv_label_pred) < _k:
                        if coerce == "abandon":  # pragma: no cover
                            continue
                        elif coerce == "raise":  # pragma: no cover
                            raise ValueError("Not enough value: %s vs target %s" % (len(inv_label_pred), _k))
                        elif coerce == "padding":
                            inv_label_pred += [(0, pad_pred)] * (_k - len(inv_label_pred))
                    k_label_pred = inv_label_pred[:_k]
                    total_label = len(label) - sum(label)

                if not k_label_pred:  # pragma: no cover
                    continue
                k_label, k_pred = list(zip(*k_label_pred))
                if len(k_label) == 1:
                    k_results[_k]["ndcg@k%s" % _suffix].append(1)
                else:
                    k_results[_k]["ndcg@k%s" % _suffix].append(ndcg_score([k_label], [k_pred]))
                p = sum(k_label) / len(k_label)
                r = sum(k_label) / total_label if total_label else 0
                k_results[_k]["precision@k%s" % _suffix].append(p)
                k_results[_k]["recall@k%s" % _suffix].append(r)
                k_results[_k]["f1@k%s" % _suffix].append(2 * p * r / (p + r) if p + r else 0)
                k_results[_k]["len@k%s" % _suffix].append(len(k_label))
                k_results[_k]["support@k%s" % _suffix].append(1)

    ret = OrderedDict()
    for key, value in results.items():
        if value:
            if key == "support":
                ret[key] = np.sum(value)
            else:
                ret[key] = np.mean(value)
    for k, key_value in k_results.items():
        ret[k] = OrderedDict()
        for key, value in key_value.items():
            if value:
                if key in {"support@k", "support@k(B)"}:
                    ret[k][key] = np.sum(value)
                else:
                    ret[k][key] = np.mean(value)
    return ret
