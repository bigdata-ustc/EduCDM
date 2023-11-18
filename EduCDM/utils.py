# coding: utf-8
# 2023/11/5 @ Fei Wang

import numpy as np


def re_index(meta_data):
    """
    Reindex the values in meta_data with continuous positive integers starting from 0
    Args:
        meta_data: dict
    Return:
        a dict with the same keys as meta_data, where the values are replaced with dicts that project
         original values to new indexes
    """
    ret = {}
    for key in meta_data.keys():
        val_arr = meta_data[key]
        if isinstance(val_arr, list):
            val_arr = np.array(val_arr)
        assert len(val_arr.shape) == 1
        val_arr = np.unique(val_arr)
        val2index = {val: i for i, val in enumerate(val_arr)}
        ret[key] = val2index
    return ret
