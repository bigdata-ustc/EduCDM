# coding: utf-8

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .utils import pack_batch, multi_hot
from longling import iterwrap
from baize.utils import pad_sequence


class Dict2(object):
    def __init__(self):
        self.u2i = {}
        self.i2u = {}

        self.u2i_r_dis = {}
        self.i2u_r_dis = {}

        self.u2i_i = {}
        self.u2i_r = {}
        self.i2u_u = {}
        self.i2u_r = {}

    def add_item_users_responses(self, iid: int, users: pd.Series,
                                 responses: pd.Series):
        if iid not in self.i2u_u:
            self.i2u_u[iid] = []
            self.i2u_r[iid] = []
            self.i2u_r_dis[iid] = [0, 0]

        for uid, r in zip(users, responses):
            self.i2u_u[iid].append(uid)
            self.i2u_r[iid].append(r)
            idx = 0 if r >= 0.5 else 1
            self.i2u_r_dis[iid][idx] += 1

    def add_user_items_responses(self, uid: int, items: pd.Series,
                                 responses: pd.Series):
        if uid not in self.u2i_i:
            self.u2i_i[uid] = []
            self.u2i_r[uid] = []
            self.u2i_r_dis[uid] = [0, 0]

        for iid, r in zip(items, responses):
            self.u2i_i[uid].append(iid)
            self.u2i_r[uid].append(r)
            idx = 0 if r >= 0.5 else 1
            self.u2i_r_dis[uid][idx] += 1

    def merge_u2i(self, new):
        merge_dict(self.u2i, new)

    def merge_i2u(self, new):
        merge_dict(self.i2u, new)

    def merge_u2i_r(self, inc_dict2):
        for uid in inc_dict2.u2i_r:
            if uid not in self.u2i_r:
                self.u2i_i[uid] = []
                self.u2i_r[uid] = []
                self.u2i_r_dis[uid] = [0, 0]
            self.u2i_i[uid].extend(inc_dict2.u2i_i[uid])
            self.u2i_r[uid].extend(inc_dict2.u2i_r[uid])
            self.u2i_r_dis[uid][0] += inc_dict2.u2i_r_dis[uid][0]
            self.u2i_r_dis[uid][1] += inc_dict2.u2i_r_dis[uid][1]

    def merge_i2u_r(self, inc_dict2):
        for iid in inc_dict2.i2u_r:
            if iid not in self.i2u_r:
                self.i2u_u[iid] = []
                self.i2u_r[iid] = []
                self.i2u_r_dis[iid] = [0, 0]
            self.i2u_u[iid].extend(inc_dict2.i2u_u[iid])
            self.i2u_r[iid].extend(inc_dict2.i2u_r[iid])
            self.i2u_r_dis[iid][0] += inc_dict2.i2u_r_dis[iid][0]
            self.i2u_r_dis[iid][1] += inc_dict2.i2u_r_dis[iid][1]


def item2knowledge(filepath, k_offset=1):  # pragma: no cover
    df_item = pd.read_csv(filepath)
    item2knowledge = {}
    knowledge_set = set()
    for i, item_know in df_item.iterrows():
        item_id, knowledge_codes = item_know['item_id'], [
            k - k_offset for k in set(eval(item_know['knowledge_code']))
        ]
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    return item2knowledge


def user2items(df: pd.DataFrame, dict2: Dict2 = None):
    grouped = df.groupby(['user_id'])
    ul_dict = {}
    if dict2:
        dict2.u2i = ul_dict
    for gp in tqdm(grouped, 'group user'):
        rs = gp[1]['item_id'] * 2 + gp[1]['score'] + 1
        if dict2:
            dict2.add_user_items_responses(gp[0], gp[1]['item_id'],
                                           gp[1]['score'])
        rs = [int(x) for x in list(rs)]
        ul_dict[gp[0]] = rs

    return ul_dict


def item2users(df: pd.DataFrame, dict2: Dict2 = None):
    grouped = df.groupby(['item_id'])
    il_dict = {}
    for gp in tqdm(grouped, 'group item'):
        rs = gp[1]['user_id'] * 2 + gp[1]['score'] + 1
        if dict2:
            dict2.add_item_users_responses(gp[0], gp[1]['user_id'],
                                           gp[1]['score'])
        rs = [int(x) for x in list(rs)]
        il_dict[gp[0]] = rs

    return il_dict


def merge_dict(src: dict, to_add: dict):
    for k, v in to_add.items():
        if k not in src:
            src[k] = []
        src[k].extend(v)


@iterwrap()
def transform(logs_df: pd.DataFrame,
              u2i,
              i2u,
              i2k,
              kn,
              batch_size,
              max_u2i=None,
              max_i2u=None,
              silent=False,
              desc="",
              allow_missing="skip",
              user_set: set = None,
              item_set: set = None):
    random_state = np.random.default_rng(0)
    batch = []
    for _, logs in tqdm(logs_df.iterrows(),
                        "batchify %s" % desc,
                        disable=silent):
        user_id, item_id, score = logs["user_id"], logs["item_id"], logs[
            "score"]
        if (user_set
                and user_id not in user_set) or (item_set
                                                 and item_id not in item_set):
            continue  # pragma: no cover
        if user_id not in u2i or item_id not in i2u:  # pragma: no cover
            if allow_missing == "skip":
                continue
            elif allow_missing is False:
                raise KeyError()
            else:
                _u2i = u2i.get(user_id, [0])
                _i2u = i2u.get(item_id, [0])
        else:
            _u2i = u2i[user_id]
            _i2u = i2u[item_id]
        batch.append([
            user_id, _u2i if max_u2i is None else random_state.choice(
                _u2i, max_u2i).tolist(),
            item_id, _i2u if max_i2u is None else random_state.choice(
                _i2u, max_i2u).tolist(),
            multi_hot(i2k[item_id], kn), score
        ])
        if len(batch) == batch_size:
            yield pack_batch(batch)
            # batch_data.append(pack_batch(batch))
            batch = []

    if batch:
        yield pack_batch(batch)
    #     batch_data.append(pack_batch(batch))
    #
    # return batch_data


def extract(filepath,
            item2know_filepath,
            dict2: Dict2 = None):  # pragma: no cover
    df = pd.read_csv(filepath)
    i2k = item2knowledge(item2know_filepath)
    u2i = user2items(df, dict2)
    i2u = item2users(df, dict2)
    return df, u2i, i2u, i2k


def test_etl(filepath,
             u2i,
             i2u,
             i2k,
             kn,
             batch_size,
             allow_missing=True):  # pragma: no cover
    logs_df = pd.read_csv(filepath)
    return transform(logs_df,
                     u2i,
                     i2u,
                     i2k,
                     kn,
                     batch_size,
                     desc=filepath,
                     allow_missing=allow_missing)


def inc_stream(logs_df: pd.DataFrame, stream_size):
    for i in range(0, logs_df.shape[0], stream_size):
        if len(logs_df[i:i + stream_size]) >= 0.9 * stream_size:
            yield logs_df[i:i + stream_size]


def dict_etl(keys, obj: dict, batch_size, silent=True):
    def pack_batch(_batch):
        __id, __records = zip(*_batch)
        length = [len(d) for d in __records]
        return (
            torch.tensor(__id, dtype=torch.int64),
            torch.tensor(pad_sequence(__records), dtype=torch.int64),
            torch.tensor(length, dtype=torch.int64),
        )

    batch_data = []
    batch = []
    for key in tqdm(keys, "dict batchfying", disable=silent):
        batch.append((key, obj[key]))
        if len(batch) == batch_size:
            batch_data.append(pack_batch(batch))
            batch = []
    if batch:
        batch_data.append(pack_batch(batch))

    return batch_data
