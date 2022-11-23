# coding: utf-8

import logging
from baize import config_logging
import os
from EduCDM.ICD.etl import extract, inc_stream
from longling import build_dir
from longling.lib.stream import to_io_group, close_io
from EduCDM.ICD.ICD import ICD

path_prefix = os.path.abspath('.')


def run(cdm,
        user_n,
        item_n,
        know_n,
        dataset,
        max_u2i=None,
        max_i2u=None,
        stream_num=50,
        alpha=0.999,
        beta=0.95,
        tolerance=1e-3,
        inner_metrics=True,
        hyper_tag=False,
        epoch=1,
        wfs=None,
        logger=logging,
        log_file="log",
        warmup_ratio=0.1,
        epsilon=1e-2,
        weight_decay=0,
        vector_numbers=None,
        vector_path_format=None,
        ctx="cpu",
        *args,
        **kwargs):
    dataset_dir = "%s/data/%s/" % (path_prefix, dataset)
    data_dir = dataset_dir
    item2know = "%sitem.csv" % dataset_dir
    path_format = "%s{}.csv" % data_dir

    inc_train_data_path = path_format.format(log_file)
    inc_train_df, _, _, i2k = extract(inc_train_data_path, item2know)
    inc_train_df_list = list(
        inc_stream(inc_train_df,
                   stream_size=int(len(inc_train_df) // stream_num)))
    ICDNet = ICD(cdm, user_n, item_n, know_n, epoch, weight_decay,
                 inner_metrics, logger, alpha, ctx)
    ICDNet.train(inc_train_df_list, i2k, beta, warmup_ratio, tolerance,
                 max_u2i, max_i2u, hyper_tag, vector_numbers,
                 vector_path_format, wfs)


def main(dataset="a0910",
         ctx="cpu",
         cdm="mirt",
         alpha=0.2,
         beta=0.9,
         tolerance=2e-1,
         epoch=1,
         pretrained=False,
         savename=None,
         inc_epoch=None,
         inner_metrics=True,
         log_file="log",
         warmup_ratio=0.1,
         epsilon=1e-2,
         stream_num=None,
         vector_numbers=None):
    if savename:
        dataset_dir = "%s/data/%s/" % (path_prefix, dataset)
        data_dir = dataset_dir
        model_dir = data_dir + "model/%s/%s/" % (cdm, savename)
        keys = [
            "metrics", "before_metrics", "ind_inc_user", "ind_inc_item",
            "inc_user", "inc_item", "new_user", "new_item", "new_both",
            "trait", "inc_trait", "tp"
        ]
        path_format = model_dir + "{}.json"
        wfs = dict(
            zip(
                keys,
                to_io_group(*[path_format.format(key) for key in keys],
                            mode="w"))) if savename else None
        logger = config_logging(model_dir + "log.txt",
                                logger="ICD",
                                console_log_level="info")
        logger.info("logs to %s" % model_dir + "log.txt")
        vector_path_format = model_dir + "{}_{}.pt"
        build_dir(vector_path_format)
    else:
        wfs = None
        logger = config_logging(logger="ICD", console_log_level="info")
        vector_path_format = None

    config = dict(
        dataset=dataset,
        cdm=cdm,
        alpha=alpha,
        beta=beta,
        tolerance=tolerance,
        ctx=ctx,
        epoch=epoch,
        inc_epoch=inc_epoch,
        inner_metrics=inner_metrics,
        log_file=log_file,
        warmup_ratio=warmup_ratio,
        epsilon=epsilon,
        vector_numbers=vector_numbers,
        vector_path_format=vector_path_format,
    )
    logger.info(config)

    dataset_config = {
        "a0910":
        dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            stream_num=50 if stream_num is None else stream_num,
            max_u2i=128,
            max_i2u=64,
        ),
        "math":
        dict(
            user_n=10269,
            item_n=17747,
            know_n=1488,
            stream_num=200 if stream_num is None else stream_num,
            # max_u2i=128,
            # max_i2u=64,
        ),
        "xunfei":
        dict(
            # user_n=10269+1,
            # item_n=2507+1,
            user_n=6820 + 1,
            item_n=1196 + 1,
            know_n=497,
            stream_num=50 if stream_num is None else stream_num,
            max_u2i=128,
            max_i2u=64,
        ),
    }
    cdm_config = {
        "irt": {},
        "dina": {},
        "ncd": {},
        "mirt": {
            "weight_decay": 1e-4
        }
    }
    run(
        # cdm="mirt",
        pretrained=pretrained,
        wfs=wfs,
        logger=logger,
        **cdm_config[cdm],
        **config,
        **dataset_config[dataset.split("_")[0]])
    if wfs is not None:
        close_io(list(wfs.values()))


if __name__ == '__main__':
    import fire

    fire.Fire(main)
