# coding: utf-8

import torch
from baize.utils import pad_sequence
from torch import Tensor, LongTensor


def multi_hot(ks, kn):
    array = [0] * kn
    for k in ks:
        array[k] = 1
    return array


def pack_batch(batch):
    user_id, user_items, item_id, item_users, item_knows, response = zip(*batch)
    user_items_length = [len(d) for d in user_items]
    padded_user_items = pad_sequence(user_items)
    item_users_length = [len(d) for d in item_users]
    padded_item_users = pad_sequence(item_users)
    return (
        LongTensor(user_id), LongTensor(padded_user_items), LongTensor(user_items_length),
        LongTensor(item_id), LongTensor(padded_item_users), LongTensor(item_users_length), Tensor(item_knows),
        Tensor(response)
    )
