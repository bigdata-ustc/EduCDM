# coding: utf-8

from baize import config_logging
from EduCDM.ICD.etl import item2knowledge, inc_stream
from EduCDM.ICD.ICD import ICD
import pandas as pd
import os

path_prefix = os.path.abspath('.')


def main(dataset="a0910",
         device="cpu",
         cdm="ncd",
         alpha=0.2,
         beta=0.9,
         tolerance=2e-1,
         epoch=1,
         inner_metrics=True,
         warmup_ratio=0.1,
         max_u2i=128,
         max_i2u=64,
         stream_num=10,
         wfs=None):

    logger = config_logging(logger="ICD", console_log_level="info")
    weight_decay = 0
    hyper_tag = False

    # load data
    train_data = pd.read_csv(f"{path_prefix}/data/{dataset}/train.csv")
    test_data = pd.read_csv(f"{path_prefix}/data/{dataset}/test.csv")
    df_item = pd.read_csv(f"{path_prefix}/data/{dataset}/item.csv")
    train_data = train_data.rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'score': 'response'})
    test_data = test_data.rename(columns={'user_id': 'userId', 'item_id': 'itemId', 'score': 'response'})
    df_item = df_item.rename(columns={'item_id': 'itemId', 'knowledge_code': 'skill'})
    train_data = pd.merge(train_data, df_item, how='left', on='itemId')
    test_data = pd.merge(test_data, df_item, how='left', on='itemId')

    knowledge_set, item_set = set(), set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['itemId'], list(set(eval(s['skill'])))
        knowledge_set.update(knowledge_codes)
        item_set.add(item_id)
    userIds = train_data['userId'].unique()
    meta_data = {'userId': list(userIds), 'itemId': list(item_set), 'skill': list(knowledge_set)}
    
    # Initialize model and training
    ICDNet = ICD(cdm, meta_data, epoch, weight_decay, inner_metrics, logger, alpha, device)
    ICDNet.fit(train_data, df_item, stream_num, beta, warmup_ratio, tolerance, max_u2i, max_i2u, hyper_tag, wfs)
    
    # save model
    ICDNet.save("icd.snapshot")

    # load model
    ICDNet.load("icd.snapshot")

    # predict using the trained model
    print(ICDNet.predict(test_data)[:10])


if __name__ == '__main__':
    import fire

    fire.Fire(main)
