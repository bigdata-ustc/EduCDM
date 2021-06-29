# coding: utf-8
# 2021/6/19 @ tongshiwei

from tqdm import tqdm
import os
import pandas as pd
from longling import print_time
import numpy as np


def extract_item(data_src, knowledge_num, params):
    with print_time("loading data from %s" % os.path.abspath(data_src), params.logger):
        knowledge = {}
        for record in tqdm(pd.read_csv(data_src).to_dict("records"), "reading records from %s" % data_src):
            knowledge_code_vector = [0] * knowledge_num
            for code in eval(record["knowledge_code"]):
                assert code >= 1
                knowledge_code_vector[code - 1] = 1
            knowledge[record["item_id"]] = np.asarray(knowledge_code_vector)
        return knowledge
