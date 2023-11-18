# coding: utf-8
# 2023/11/5 @ Fei Wang

import numpy as np


def re_index(meta_data):
    """
    Reindex the values in meta_data with continuous positive integers starting from 0
    Args:
        meta_data: dict
    Return:
        tuple (ret_val2index, ret_index2val). Two dictionaries with the same keys as meta_data, where ret_val2index
         replaces the original values with dicts that map original values to new indexes. ret_index2val stores the
         reverse mapping.
    """
    ret_val2index, ret_index2val = {}, {}
    for key in meta_data.keys():
        val_arr = meta_data[key]
        if isinstance(val_arr, list):
            val_arr = np.array(val_arr)
        assert len(val_arr.shape) == 1
        val_arr = np.unique(val_arr)
        val2index = {val: i for i, val in enumerate(val_arr)}
        index2val = {i: val for i, val in enumerate(val_arr)}
        ret_val2index[key] = val2index
        ret_index2val[key] = index2val
    return ret_val2index, ret_index2val
