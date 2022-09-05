# coding: utf-8

from longling import build_dir
import logging
from copy import deepcopy
import torch
import pandas as pd
from baize.torch import Configuration
from baize.torch import light_module as lm
from longling.lib.stream import to_io_group, close_io
from ICD.etl import inc_stream
from etl import extract, transform, etl, item2knowledge
from sym import fit_f, eval_f, get_loss, get_net, stableness_eval
from baize import config_logging
from ICD.utils import output_metrics
from ICD.constant import path_prefix

def run(user_n, item_n, know_n, dataset, cdm, inc_type=None, stream_num=50, wfs=None, logger=logging,
        log_file="log", warmup_ratio=0.1, weight_decay=0, vector_numbers=None, inner_metrics=False,
        vector_path_format=None, *args, **kwargs):

    torch.manual_seed(0)

    dataset_dir = "%s/data/%s/"%(path_prefix,dataset)
    data_dir = dataset_dir

    cfg = Configuration(
        model_name="%s" % cdm,
        model_dir="%s" % cdm,
        end_epoch=1,
        batch_size=32,
        hyper_params={"user_num": user_n, "item_num": item_n, "know_n": know_n},
        # train_select={".*int.*": {'weight_decay': 0}, "^(?!.*int)": {}},
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            'weight_decay': weight_decay
        },
        ctx=kwargs.get("ctx", "cuda: 3"),
        time_digital=True
    )
    logger.info(cfg)

    item2know = "%sitem.csv" % dataset_dir
    path_format = "%s{}.csv" % data_dir

    inc_train_data_path = path_format.format(log_file)

    i2k = item2knowledge(item2know)

    net = get_net(ctx=cfg.ctx, cdm=cdm, **cfg.hyper_params)
    loss_f = get_loss(ctx=cfg.ctx)

        # stream inc
    inc_train_df = extract(inc_train_data_path)
    inc_train_df_list = list(inc_stream(inc_train_df, stream_size=int(len(inc_train_df) // stream_num)))

    vector_user = None
    vector_item = None
    user = set()
    item = set()
    train_df = pd.DataFrame()
    warmup = warmup_ratio * len(inc_train_df_list)

    # warmup = 5
    for i, inc_train_df in enumerate(inc_train_df_list):
        if i + 1 == len(inc_train_df_list):
            break

        pre_user = list(user)
        pre_item = list(item)
        pre_net = deepcopy(net)
        inc_user = set(inc_train_df["user_id"].tolist())
        inc_item = set(inc_train_df["item_id"].tolist())
        new_user = inc_user - user
        new_item = inc_item - item
        user |= inc_user
        item |= inc_item

        if i == 0:
            vector_user = list(user)
            vector_item = list(item)

        logger.info("============= Stream[%s/%s] =============" % (i, len(inc_train_df_list)))

        train_df = pd.concat([train_df, inc_train_df])
        if inc_type == "global" or i < warmup:
            inc_train_data = transform(train_df, i2k, know_n, cfg.batch_size)
        else:
            inc_train_data = transform(inc_train_df, i2k, know_n, cfg.batch_size)
        
        # if i>46 or i+2 ==len(inc_train_df_list):
        #     inc_test_data=transform(
        #         inc_train_df_list[i+1],
        #         i2k,know_n,cfg.batch_size,
        #         user_set=user,item_set=item
        #     )
        #     cfg.end_epoch=1
        #     lm.train(
        #         net=net,
        #         cfg=cfg,
        #         loss_function=loss_f,
        #         trainer=None,
        #         train_data=inc_train_data,
        #        	test_data=inc_test_data,
        #         fit_f=fit_f,
        #         eval_f=eval_f,
        #         initial_net=False
        #     )
        #     continue
        lm.train(
            net=net,
            cfg=cfg,
            loss_function=loss_f,
            trainer=None,
            train_data=inc_train_data,
            fit_f=fit_f,
            eval_f=eval_f,
            initial_net=False,
        )
        # if i % max(round(len(inc_train_df_list) // 10), 1) == 0:
        # 最后两轮开始训练 global
        if i + 2 == len(inc_train_df_list) or inner_metrics:

            inc_test_data = transform(
                inc_train_df_list[i + 1],
                i2k, know_n, cfg.batch_size,
                user_set=user, item_set=item
            )
            # inc_user_test_data = transform(
            #     inc_train_df_list[i + 1],
            #     i2k, know_n, cfg.batch_size,
            #     user_set=new_user, item_set=item
            # )
            # now_inc_user_test_data = transform(
            #     train_df,
            #     i2k, know_n, cfg.batch_size,
            #     user_set=new_user
            # )
            # inc_item_test_data = transform(
            #     inc_train_df_list[i + 1],
            #     i2k, know_n, cfg.batch_size,
            #     user_set=user, item_set=new_item
            # )
            # now_inc_item_test_data = transform(
            #     train_df,
            #     i2k, know_n, cfg.batch_size,
            #     item_set=new_item
            # )
            # inc_ui_test_data = transform(
            #     inc_train_df_list[i + 1],
            #     i2k, know_n, cfg.batch_size,
            #     user_set=new_user, item_set=new_item
            # )

            logger.info("===================== %s valid ======================" % inc_type)
            # if inc_user_test_data:
            #     output_metrics(
            #         i,
            #         eval_f(net, inc_user_test_data),
            #         wfs,
            #         "new_user",
            #         logger
            #     )
            # if now_inc_user_test_data:
            #     output_metrics(
            #         i,
            #         eval_f(net, now_inc_user_test_data),
            #         wfs,
            #         "inc_user",
            #         logger
            #     )
            # if inc_item_test_data:
            #     output_metrics(
            #         i,
            #         eval_f(net, inc_item_test_data),
            #         wfs,
            #         "new_item",
            #         logger
            #     )
            # if now_inc_item_test_data:
            #     output_metrics(
            #         i,
            #         eval_f(net, now_inc_item_test_data),
            #         wfs,
            #         "inc_item",
            #         logger
            #     )
            # if inc_ui_test_data:
            #     output_metrics(
            #         i,
            #         eval_f(net, inc_ui_test_data),
            #         wfs,
            #         "new_both",
            #         logger
            #     )
            if inc_test_data:
                output_metrics(
                    i,
                    eval_f(net, inc_test_data),
                    wfs,
                    "metrics",
                    logger
                )

            if vector_numbers and i in vector_numbers:
                trait_net = net.module if isinstance(net, torch.nn.DataParallel) else net
                vector_user_traits = trait_net.get_user_profiles(vector_user)
                vector_item_traits = trait_net.get_item_profiles(vector_item)
                vector_path = vector_path_format.format("user", i)
                logger.info("user traits to %s" % vector_path)
                torch.save(vector_user_traits, vector_path)
                vector_path = vector_path_format.format("item", i)
                logger.info("item traits to %s" % vector_path)
                torch.save(vector_item_traits, vector_path)

            if i > 0:
                eval_net = pre_net.module if isinstance(pre_net, torch.nn.DataParallel) else pre_net
                user_traits = eval_net.get_user_profiles(pre_user)
                item_traits = eval_net.get_item_profiles(pre_item)
                output_metrics(
                    i,
                    stableness_eval(net, pre_user, pre_item, user_traits, item_traits),
                    wfs,
                    "trait",
                    logger
                )
                inc_user = list(inc_user)
                inc_item = list(inc_item)
                user_traits = eval_net.get_user_profiles(inc_user)
                item_traits = eval_net.get_item_profiles(inc_item)
                output_metrics(
                    i,
                    stableness_eval(net, inc_user, inc_item, user_traits, item_traits),
                    wfs,
                    "inc_trait",
                    logger
                )


# def main(dataset="a0910", cdm="irt", inc_type="inc", ctx="cuda:1", log_file="log", warmup_ratio=0.1,lr=0.002, savename=None):
def main(dataset="a0910", cdm="ncd", inc_type="inc", ctx="cuda:1", log_file="log", warmup_ratio=0.1, lr=0.002,savename=None,
         vector_numbers=None):

    if savename:
        # /home/yutingh/icd/data/a0910
        dataset_dir = "/home/yutingh/icd/data/%s/" % dataset
        data_dir = dataset_dir
        model_dir = data_dir + "model/%s/%s/" % (cdm, savename)
        keys = [
            "metrics",
            "inc_user", "inc_item",
            "new_user", "new_item", "new_both",
            "trait", "inc_trait"
        ]
        path_format = model_dir + "{}.json"
        wfs = dict(zip(keys, to_io_group(
            *[path_format.format(key) for key in keys], mode="w"
        ) if savename else None))
        logger = config_logging(model_dir + "log.txt", logger="Base", console_log_level="info")
        logger.info({
            "dataset": dataset,
            "cdm": cdm,
            "inc_type": inc_type,
            "log_file": log_file,
            "warmup_ratio": warmup_ratio
        })
        vector_path_format = model_dir + "traits/{}_{}.pt"
        build_dir(vector_path_format)
    else:
        wfs = None
        logger = config_logging(logger="Base", console_log_level="info")
        vector_path_format = None

    dataset_config = {
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            stream_num=50,
            # max_u2i=64,
            # max_i2u=32
        ),
        "a0910_1": dict(
            user_n=4163+1,
            item_n=17751+1,
            know_n=123,
            stream_num=50,
        ),
        # "xunfei": dict(
        #     user_n=10269+1,
        #     item_n=2507+1,
        #     know_n=497,
        #     stream_num=200,
        #     # max_u2i=128,
        #     max_i2u=512,
        # ),
        "xunfei": dict(
            # user_n=10269+1,
            # item_n=2507+1,
            user_n=6820 +1,
            item_n= 1196+1,
            know_n=497,
            stream_num=50,
            max_u2i=128,
            max_i2u=128,
        ),
    }
    cdm_config = {
        "ncd": {},
        "irt": {},
        "dina": {},
        "mirt": {"weight_decay": 1e-4}
    }
    run(
        cdm=cdm,
        dataset=dataset,
        inc_type=inc_type,
        ctx=ctx,
        wfs=wfs,
        logger=logger,
        log_file=log_file,
        warmup_ratio=warmup_ratio,
        lr=lr,
        **cdm_config[cdm],
        **dataset_config[dataset.split("_")[0]]
    )
    if wfs:
        close_io(list(wfs.values()))


if __name__ == '__main__':
    import fire

    fire.Fire(main)
