# coding: utf-8

from baize import config_logging
from EduCDM.ICD.etl import item2knowledge, inc_stream
from EduCDM.ICD.ICD import ICD
import pandas as pd
import os

path_prefix = os.path.abspath('.')


def main(dataset="a0910",
         ctx="cpu",
         cdm="ncd",
         alpha=0.2,
         beta=0.9,
         tolerance=2e-1,
         epoch=1,
         inner_metrics=True,
         log_file="train",
         warmup_ratio=0.1,
         max_u2i=128,
         max_i2u=64,
         stream_num=50,
         wfs=None):

    logger = config_logging(logger="ICD", console_log_level="info")

    weight_decay = 0
    hyper_tag = False
    train_data = pd.read_csv(f"{path_prefix}/data/{dataset}/{log_file}.csv")
    df_item = pd.read_csv(f"{path_prefix}/data/{dataset}/item.csv")
    train_data = train_data.rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'score': 'response'})
    df_item = df_item.rename(columns={'item_id': 'itemId', 'knowledge_code': 'skill'})
    i2k = item2knowledge(df_item)

    knowledge_set, item_set = set(), set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['itemId'], list(set(eval(s['skill'])))
        knowledge_set.update(knowledge_codes)
        item_set.add(item_id)
    userIds = train_data['userId'].unique()
    meta_data = {'userId': list(userIds), 'itemId': list(item_set), 'skill': list(knowledge_set)}
    train_data = pd.merge(train_data, df_item, how='left', on='itemId')
    inc_train_df_list = list(
        inc_stream(train_data, stream_size=int(len(train_data) // stream_num)))
    ICDNet = ICD(cdm, meta_data, epoch, weight_decay, inner_metrics, logger, alpha, ctx)
    ICDNet.fit(inc_train_df_list, i2k, beta, warmup_ratio, tolerance, max_u2i, max_i2u, hyper_tag, wfs)
    

if __name__ == '__main__':
    import fire

    fire.Fire(main)
