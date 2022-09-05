# coding: utf-8

import logging
import pandas as pd
from copy import deepcopy
import torch
from baize.torch import Configuration
from baize.torch import light_module as lm
from baize import config_logging

from ICD.etl import extract, transform, inc_stream, user2items, item2users, dict_etl, Dict2
from sym import eval_f, get_net, DualICD, get_dual_loss, dual_fit_f, stableness_eval, turning_point
from longling import build_dir
from longling.lib.stream import to_io_group, close_io
from ICD.utils import output_metrics
from ICD.constant import path_prefix

def run(cdm, user_n, item_n, know_n, dataset, max_u2i=None, max_i2u=None,
        stream_num=50, alpha=0.999, beta=0.95, tolerance=1e-3,
        reporthook=None, final_reporthook=None, inner_metrics=True,
        hyper_tag=False, epoch=1, wfs=None, logger=logging, log_file="log", warmup_ratio=0.1,
        epsilon=1e-2, weight_decay=0, vector_numbers=None, vector_path_format=None,
        *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "%s/data/%s/" % (path_prefix,dataset)
    data_dir = dataset_dir

    cfg = Configuration(
        model_name="icd_%s" % cdm,
        model_dir="icd_%s" % cdm,
        end_epoch=epoch,
        batch_size=32,
        hyper_params={"user_n": user_n, "item_n": item_n, "know_n": know_n, "cdm": cdm},
        # train_select={".*dtn.*": {}, "^(?!.*dtn)": {'weight_decay': 0}},
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            'weight_decay': weight_decay
        },
        ctx=kwargs.get("ctx", "cuda: 0"),
        time_digital=True,
    )
    logger.info(cfg)

    item2know = "%sitem.csv" % dataset_dir
    path_format = "%s{}.csv" % data_dir

    inc_train_data_path = path_format.format(log_file)
    inc_train_df, _, _, i2k = extract(inc_train_data_path, item2know)
    inc_train_df_list = list(inc_stream(inc_train_df, stream_size=int(len(inc_train_df) // stream_num)))

    net = get_net(**cfg.hyper_params, ctx=cfg.ctx)

    vector_user = None
    vector_item = None
    dict2 = Dict2()
    act_dual_loss_f = get_dual_loss(cfg.ctx, beta=beta)
    warmup_dual_loss_f = get_dual_loss(cfg.ctx, beta=1)
    dual_net = DualICD(deepcopy(net), net, alpha=alpha)
    tps = []
    warmup = int(warmup_ratio * len(inc_train_df_list))
    train_df = pd.DataFrame()
    for i, inc_train_df in enumerate(inc_train_df_list):
        if i + 1 == len(inc_train_df_list):
            break
        if i <= warmup:
            dual_loss_f = warmup_dual_loss_f
        else:
            dual_loss_f = act_dual_loss_f

        pre_dict2 = deepcopy(dict2)
        inc_dict2 = Dict2()
        inc_u2i = user2items(inc_train_df, inc_dict2)
        inc_i2u = item2users(inc_train_df, inc_dict2)
        dual_net.stat_net = deepcopy(dual_net.net)

        logger.info("============= Stream[%s/%s/%s] =============" % (i, len(tps), len(inc_train_df_list)))
        pre_net = deepcopy(net)
        pre_net.eval()

        if i < warmup or turning_point(net, inc_train_df, dict2, inc_dict2, i2k, know_n, cfg.batch_size, ctx=cfg.ctx,
                                       tolerance=tolerance, logger=logger):
            logger.info("**** Turning Point ****")
            tps.append(i)

            dict2.merge_u2i(inc_u2i)
            dict2.merge_i2u(inc_i2u)
            if i < warmup:
                inc_train_df = train_df = pd.concat([train_df, inc_train_df])
            inc_train_data = transform(
                inc_train_df, dict2.u2i, dict2.i2u, i2k, know_n,
                max_u2i=max_u2i, max_i2u=max_i2u,
                batch_size=cfg.batch_size, silent=True
            )
            lm.train(
                net=dual_net,
                cfg=cfg,
                loss_function=dual_loss_f,
                trainer=None,
                train_data=inc_train_data,
                fit_f=dual_fit_f,
                eval_f=eval_f,
                initial_net=False,
                verbose=not hyper_tag,
            )
            if i > warmup:
                dual_net.momentum_weight_update(pre_net, cfg.train_select)
            dual_net.stat_net = pre_net

        else:
            dict2.merge_u2i(inc_u2i)
            dict2.merge_i2u(inc_i2u)

        dict2.merge_u2i_r(inc_dict2)
        dict2.merge_i2u_r(inc_dict2)

        if i == 0:
            vector_user = list(dict2.u2i.keys())
            vector_item = list(dict2.i2u.keys())

        if vector_numbers and i in vector_numbers:
            _net = dual_net.net
            trait_net = _net.module if isinstance(_net, torch.nn.DataParallel) else _net
            vector_user_traits = trait_net.get_user_profiles(
                dict_etl(vector_user, pre_dict2.u2i, batch_size=cfg.batch_size)
            )
            vector_item_traits = trait_net.get_item_profiles(
                dict_etl(vector_item, pre_dict2.i2u, batch_size=cfg.batch_size)
            )
            vector_path = vector_path_format.format("user", i)
            logger.info("user traits to %s" % vector_path)
            torch.save(vector_user_traits, vector_path)
            vector_path = vector_path_format.format("item", i)
            logger.info("item traits to %s" % vector_path)
            torch.save(vector_item_traits, vector_path)

        if i + 2 == len(inc_train_df_list) or inner_metrics:
            # new_user = set(inc_u2i.keys()) - set(pre_dict2.u2i.keys())
            # new_item = set(inc_i2u.keys()) - set(pre_dict2.i2u.keys())
            inc_test_data = transform(
                inc_train_df_list[i + 1], dict2.u2i, dict2.i2u, i2k, know_n,
                max_u2i=max_u2i, max_i2u=max_i2u,
                batch_size=cfg.batch_size, silent=True
            )
            # inc_user_test_data = transform(
            #     inc_train_df_list[i + 1], dict2.u2i, dict2.i2u, i2k, know_n,
            #     max_u2i=max_u2i, max_i2u=max_i2u,
            #     batch_size=cfg.batch_size, silent=True, user_set=new_user
            # )
            # now_inc_user_test_data = transform(
            #     train_df, dict2.u2i, dict2.i2u, i2k, know_n,
            #     max_u2i=max_u2i, max_i2u=max_i2u,
            #     batch_size=cfg.batch_size, silent=True, user_set=new_user
            # )
            # inc_item_test_data = transform(
            #     inc_train_df_list[i + 1], dict2.u2i, dict2.i2u, i2k, know_n,
            #     max_u2i=max_u2i, max_i2u=max_i2u,
            #     batch_size=cfg.batch_size, silent=True, item_set=new_item
            # )
            # now_inc_item_test_data = transform(
            #     train_df, dict2.u2i, dict2.i2u, i2k, know_n,
            #     max_u2i=max_u2i, max_i2u=max_i2u,
            #     batch_size=cfg.batch_size, silent=True, user_set=new_item
            # )
            # inc_both_test_data = transform(
            #     inc_train_df_list[i + 1], dict2.u2i, dict2.i2u, i2k, know_n,
            #     max_u2i=max_u2i, max_i2u=max_i2u,
            #     batch_size=cfg.batch_size, silent=True, user_set=new_user, item_set=new_item
            # )
            # inc_user_met = eval_f(net, inc_user_test_data)
            # output_metrics(i, inc_user_met, wfs, "new_user", logger)
            # inc_item_met = eval_f(net, inc_item_test_data)
            # output_metrics(i, inc_item_met, wfs, "new_item", logger)
            # inc_both_met = eval_f(net, inc_both_test_data)
            # output_metrics(i, inc_both_met, wfs, "new_both", logger)

            # before_inc_user_met = eval_f(pre_net, now_inc_user_test_data)
            # output_metrics(i, before_inc_user_met, wfs, "ind_inc_user", logger)
            # now_inc_user_met = eval_f(net, now_inc_user_test_data)
            # output_metrics(i, now_inc_user_met, wfs, "inc_user", logger)
            # before_inc_item_met = eval_f(pre_net, now_inc_item_test_data)
            # output_metrics(i, before_inc_item_met, wfs, "ind_inc_item", logger)
            # now_inc_item_met = eval_f(net, now_inc_user_test_data)
            # output_metrics(i, now_inc_item_met, wfs, "inc_item", logger)

            # before_inc_met = eval_f(pre_net, inc_test_data)
            # output_metrics(i, before_inc_met, wfs, "before_metrics", logger)
            inc_met = eval_f(net, inc_test_data)
            output_metrics(i, inc_met, wfs, "metrics", logger)
            if i > 0:
                _net = dual_net.stat_net
                stat_net = _net.module if isinstance(_net, torch.nn.DataParallel) else _net

                users = list(pre_dict2.u2i.keys())
                items = list(pre_dict2.i2u.keys())
                user_traits = stat_net.get_user_profiles(dict_etl(users, pre_dict2.u2i, batch_size=cfg.batch_size))
                item_traits = stat_net.get_item_profiles(dict_etl(items, pre_dict2.i2u, batch_size=cfg.batch_size))
                sta_met = stableness_eval(dual_net.net, users, items, pre_dict2.u2i, pre_dict2.i2u, user_traits,
                                          item_traits,
                                          cfg.batch_size)

                inc_users = list(inc_u2i.keys())
                inc_items = list(inc_i2u.keys())
                inc_user_traits = stat_net.get_user_profiles(dict_etl(inc_users, inc_u2i, batch_size=cfg.batch_size))
                inc_item_traits = stat_net.get_item_profiles(dict_etl(inc_items, inc_i2u, batch_size=cfg.batch_size))
                inc_sta_met = stableness_eval(dual_net.net, inc_users, inc_items, inc_u2i, inc_i2u,
                                              inc_user_traits, inc_item_traits, cfg.batch_size)

                output_metrics(i, sta_met, wfs, "trait", logger)
                output_metrics(i, inc_sta_met, wfs, "inc_trait", logger)

    output_metrics(0, {"tps": tps, "tp_cnt": len(tps), "total": len(inc_train_df_list) - 1}, wfs, "tp", logger)


def main(dataset="xunfei", ctx="cuda:0", cdm="mirt",
         alpha=0.2, beta=0.9, tolerance=2e-1, epoch=1, pretrained=False, savename=None,
         inc_epoch=None, inner_metrics=True, log_file="log", warmup_ratio=0.1, epsilon=1e-2, stream_num=None,
         vector_numbers=None):
    if savename:
        dataset_dir = "%s/data/%s/" % (path_prefix,dataset)
        data_dir = dataset_dir
        model_dir = data_dir + "model/%s/%s/" % (cdm, savename)
        keys = [
            "metrics", "before_metrics",
            "ind_inc_user", "ind_inc_item", "inc_user", "inc_item",
            "new_user", "new_item", "new_both",
            "trait", "inc_trait", "tp"
        ]
        path_format = model_dir + "{}.json"
        wfs = dict(zip(keys, to_io_group(
            *[path_format.format(key) for key in keys], mode="w"
        ))) if savename else None
        logger = config_logging(model_dir + "log.txt", logger="ICD", console_log_level="info")
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
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            stream_num=50 if stream_num is None else stream_num,
            max_u2i=128,
            max_i2u=64,
        ),
        "math": dict(
            user_n=10269,
            item_n=17747,
            know_n=1488,
            stream_num=200 if stream_num is None else stream_num,
            # max_u2i=128,
            # max_i2u=64,
        ),
        "xunfei": dict(
            # user_n=10269+1,
            # item_n=2507+1,
            user_n=6820 +1,
            item_n=1196+1,
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
        "mirt": {"weight_decay": 1e-4}
    }
    run(
        # cdm="mirt",
        pretrained=pretrained,
        # ctx="cpu",
        wfs=wfs,
        logger=logger,
        **cdm_config[cdm],
        **config,
        **dataset_config[dataset.split("_")[0]]
    )
    if wfs is not None:
        close_io(list(wfs.values()))


if __name__ == '__main__':
    import fire

    fire.Fire(main)
