# coding: utf-8
# 2021/6/19 @ tongshiwei
import torch
import os
from longling import print_time, iterwrap
import pandas as pd
import numpy as np

from longling.ML.toolkit.dataset import ItemSpecificSampler

__all__ = ["etl"]


def extract(data_src, params):
    with print_time("loading data from %s" % os.path.abspath(data_src), params.logger):
        df = pd.read_csv(data_src, dtype={"user_id": "int64", "item_id": "int64", "score": "float32"})
        sampler = ItemSpecificSampler(
            ItemSpecificSampler.rating2triplet(
                df, query_field="item_id", key_field="user_id", value_field="score"
            ),
            query_field="item_id", user_id_range=[1, params.hyper_params["user_num"]],
        )
        return df, sampler


@iterwrap()
def transform(raw_data, knowledge, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size
    n_neg = params.n_neg
    n_imp = params.n_imp
    df: pd.DataFrame = raw_data[0]
    sampler: ItemSpecificSampler = raw_data[1]

    for start in range(0, len(df), batch_size):
        _df = df.iloc[start: start + batch_size]
        n_sample, sample = sampler(
            _df["item_id"], n_neg, neg=_df["score"] != 0.0, return_column=True, padding=True,
            split_sample_to_column=True, verbose=False, padding_implicit=False,
            fast_implicit=True, with_n_implicit=n_imp
        ) if (n_neg + n_imp) > 0 else ([0] * _df.shape[0], [])
        _knowledge = np.stack([knowledge[int(item)] for item in _df["item_id"]]).astype("float32")
        yield [
            torch.tensor(array if not isinstance(array, pd.Series) else array.values) for array in
            [_df["user_id"], _df["item_id"], _knowledge, _df["score"],
             n_sample, *sample]
        ]


@iterwrap()
def load(transformed_data, params):
    return transformed_data


def etl(filepath, knowledge, params):
    raw_data = extract(filepath, params)
    transformed_data = transform(raw_data, knowledge, params)
    return load(transformed_data, params), raw_data[0]
