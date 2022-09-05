# coding: utf-8

import json
from copy import deepcopy
import torch
from baize.torch import Configuration
from baize.torch import light_module as lm, load_net
from baize.metrics import POrderedDict
from longling.ML.PytorchHelper import set_device

from ICD.etl import extract, transform, test_etl, merge_dict, inc_stream, user2items, item2users, dict_etl, Dict2
from sym import fit_f, eval_f, get_loss, get_net, DualICD, get_dual_loss, dual_fit_f, stableness_eval, turning_point

from longling.ML.toolkit.hyper_search import prepare_hyper_search
from longling.lib.stream import to_io_group, close_io


def format_metrics_result(_id, ori_met, inc_met=None, dt=0):
    ret = {
        "id": _id,
        "ori": {"accuracy": ori_met["accuracy"], "auc": ori_met["macro_auc"], "doa": ori_met["doa"], "deltaT": dt},

    }
    if inc_met is not None:
        ret.update({
            "inc": {"accuracy": inc_met["accuracy"], "auc": inc_met["macro_auc"], "doa": inc_met["doa"]}
        })
    return POrderedDict(ret)


def output_metrics(metrics, wfs):
    print(metrics)
    if wfs is not None:
        print(metrics, file=wfs[0], flush=True)
        print(json.dumps(metrics), file=wfs[1], flush=True)


def run(cdm, user_n, item_n, know_n, dataset, scenario, max_u2i=None, max_i2u=None, pretrained=False,
        stream_size=2048, alpha=0.999, beta=0.95, tolerance=1e-3,
        reporthook=None, final_reporthook=None, hyper_tag=False, epoch=3, wfs=None,
        *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "../../data/%s/" % dataset
    data_dir = dataset_dir + "%s/" % scenario

    cfg = Configuration(
        model_name="icd_%s" % cdm,
        model_dir="icd_%s" % cdm,
        end_epoch=epoch,
        batch_size=32,
        hyper_params={"user_n": user_n, "item_n": item_n, "know_n": know_n, "cdm": cdm},
        # train_select={".*dtn.*": {}, "^(?!.*dtn)": {'weight_decay': 0}},
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            # 'weight_decay': 0.0001
        },
        ctx=kwargs.get("ctx", "cuda: 0"),
        time_digital=True,
    )
    print(cfg)
    if wfs is not None:
        print(cfg, file=wfs[0])

    item2know = "%s/item.csv" % dataset_dir
    path_format = "%s/{}.csv" % data_dir

    stat_train_data_path = path_format.format("stat_train")
    stat_valid_data_path = path_format.format("stat_valid")
    stat_test_data_path = path_format.format("stat_test")

    inc_train_data_path = path_format.format("inc_train")
    inc_valid_data_path = path_format.format("inc_valid")
    inc_test_data_path = path_format.format("inc_test")

    dict2 = Dict2()
    train_df, u2i, i2u, i2k = extract(stat_train_data_path, item2know, dict2)
    dict2.u2i = u2i
    dict2.i2u = i2u

    stat_train_data = transform(
        train_df, u2i, i2u, i2k, know_n, cfg.batch_size,
        max_u2i=max_u2i, max_i2u=max_i2u, silent=True
    )
    stat_valid_data = list(test_etl(stat_valid_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))
    stat_test_data = list(test_etl(stat_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))

    net = get_net(**cfg.hyper_params)
    loss_f = get_loss(ctx=cfg.ctx)
    if pretrained is True:
        print("load from pretrained")
        load_net(data_dir + "model/%s/icd_%s_init.params" % (cdm, cdm), net)
        net = set_device(net, cfg.ctx)
        pre_metrics = format_metrics_result(
            "before pretrained",
            eval_f(net, stat_test_data),
        )
        output_metrics(pre_metrics, wfs)

        if cdm == "ncd":
            print("finetune int fc")
            # cfg.train_select = ".*int.*"
            # cfg.optimizer_params["lr"] = 0.0001
            cfg.end_epoch, end_epoch = kwargs.get("pre_epoch", epoch), cfg.end_epoch
            print(cfg)
            lm.train(
                net=net,
                cfg=cfg,
                loss_function=loss_f,
                trainer=None,
                train_data=stat_train_data,
                test_data=stat_valid_data,
                fit_f=fit_f,
                eval_f=eval_f,
                initial_net=False,
            )
            cfg.end_epoch = end_epoch
            print("Ori.")
            after_metrics = format_metrics_result(
                "after pretrained",
                eval_f(net, stat_test_data)
            )
            output_metrics(after_metrics, wfs)
    else:
        net = set_device(net, cfg.ctx)

    lm.train(
        net=net,
        cfg=cfg,
        loss_function=loss_f,
        trainer=None,
        train_data=stat_train_data,
        test_data=stat_test_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=False,
        verbose=not hyper_tag,
    )

    stat_inc_u2i = deepcopy(u2i)
    stat_inc_i2u = deepcopy(i2u)
    inc_train_df, inc_u2i, inc_i2u, _ = extract(inc_train_data_path, item2know)
    merge_dict(stat_inc_u2i, inc_u2i)
    merge_dict(stat_inc_i2u, inc_i2u)
    inc_test_data = list(test_etl(inc_test_data_path, stat_inc_u2i, stat_inc_i2u, i2k, know_n, cfg.batch_size))
    inc_train_df_list = list(inc_stream(inc_train_df, stream_size=stream_size))

    print("=============== Stat. ===================")

    print("Ori.")
    ori_met = eval_f(net, stat_test_data)
    print(ori_met)

    print("Inc.")
    inc_met = eval_f(net, inc_test_data)
    print(inc_met)

    if hyper_tag:
        reporthook(format_metrics_result(0, ori_met, inc_met))

    output_metrics(format_metrics_result("Stat.", ori_met, inc_met), wfs)

    stat_net = net.module if isinstance(net, torch.nn.DataParallel) else net

    users = list(u2i.keys())
    items = list(i2u.keys())
    user_traits = stat_net.get_user_profiles(dict_etl(users, u2i, batch_size=cfg.batch_size))
    item_traits = stat_net.get_item_profiles(dict_etl(items, i2u, batch_size=cfg.batch_size))

    # cfg.train_select = ".*dtn.*"
    cfg.end_epoch = kwargs.get("inc_epoch", cfg.end_epoch)
    print(cfg)

    dual_loss_f = get_dual_loss(cfg.ctx, beta=beta)
    stat_net = deepcopy(net)
    dual_net = DualICD(stat_net, net, alpha=alpha)
    tps = []
    for i, inc_train_df in enumerate(inc_train_df_list):
        inc_dict2 = Dict2()
        inc_u2i = user2items(inc_train_df, inc_dict2)
        inc_i2u = item2users(inc_train_df, inc_dict2)

        print("============= Stream[%s/%s/%s] =============" % (i, len(tps), len(inc_train_df_list)))

        if turning_point(net, inc_train_df, dict2, inc_dict2, i2k, know_n, cfg.batch_size, ctx=cfg.ctx,
                         tolerance=tolerance):
            print("**** Turning Point ****")
            tps.append(i)

            dict2.merge_u2i(inc_u2i)
            dict2.merge_i2u(inc_i2u)
            inc_train_data = transform(
                inc_train_df, u2i, i2u, i2k, know_n,
                max_u2i=max_u2i, max_i2u=max_i2u,
                batch_size=cfg.batch_size, silent=True
            )

            stat_net.eval()
            pre_net = deepcopy(net)
            pre_net.eval()
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
            dual_net.momentum_weight_update(pre_net, cfg.train_select)

        else:
            dict2.merge_u2i(inc_u2i)
            dict2.merge_i2u(inc_i2u)

        dict2.merge_u2i_r(inc_dict2)
        dict2.merge_i2u_r(inc_dict2)
        if i % max(round(len(inc_train_df_list) // 10), 1) == 0:
            # stat_valid_data = list(test_etl(stat_valid_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))
            # inc_valid_data = list(test_etl(inc_valid_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))
            #
            # print("=============== Inc. ===================")
            #
            # print("Ori.")
            # print(eval_f(net, stat_valid_data))
            #
            # print("Inc.")
            # print(eval_f(net, inc_valid_data))
            #
            # print("Trait")
            # print(stableness_eval(net, users, items, u2i, i2u, user_traits, item_traits, cfg.batch_size))

            stat_valid_data = list(
                test_etl(stat_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size, allow_missing=True))
            inc_valid_data = list(
                test_etl(inc_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size, allow_missing=True))

            print("=============== Inc. ===================")

            print("Ori.")
            ori_met = eval_f(net, stat_valid_data)
            print(ori_met)

            print("Inc.")
            inc_met = eval_f(net, inc_valid_data)
            print(inc_met)

            print("Trait")
            sta_met = stableness_eval(net, users, items, u2i, i2u, user_traits, item_traits, cfg.batch_size)
            print(sta_met)

            inner_metrics = format_metrics_result(i + 1, ori_met, inc_met, sta_met["micro_ave"])
            if hyper_tag:
                reporthook(inner_metrics)
            output_metrics(inner_metrics, wfs)

    inc_test_data = list(test_etl(inc_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))
    stat_test_data = list(test_etl(stat_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))

    print("=============== Inc. ===================")

    print("Ori.")
    ori_met = eval_f(net, stat_test_data)
    print(ori_met)

    print("Inc.")
    inc_met = eval_f(net, inc_test_data)
    print(inc_met)

    print("Trait")
    sta_met = stableness_eval(net, users, items, u2i, i2u, user_traits, item_traits, cfg.batch_size)
    print(sta_met)

    final_metrics = format_metrics_result(len(inc_train_df_list), ori_met, inc_met, sta_met["micro_ave"])
    if hyper_tag:
        final_reporthook(final_metrics)
    output_metrics(final_metrics, wfs)

    print("TP %s/%s" % (len(tps), len(inc_train_df_list)))
    if wfs is not None:
        print("TP %s/%s" % (len(tps), len(inc_train_df_list)), file=wfs[0], flush=True)
        print("TP: %s" % tps, file=wfs[0], flush=True)


def main(dataset="a0910", scenario="new_user", ctx="cuda: 3", cdm="ncd",
         stream_size=2048, alpha=0.999, beta=0.95, tolerance=1e-2, epoch=3, pretrained=False, filename=None,
         inc_epoch=None):
    dataset_dir = "../../data/%s/" % dataset
    data_dir = dataset_dir + "%s/" % scenario
    model_dir = data_dir + "model/%s/" % cdm
    wfs = to_io_group(
        model_dir + filename + ".log",
        model_dir + filename + ".json",
        mode="w"
    ) if filename else None

    # if wfs is not None:
    #     close_io(wfs)
    # wfs = to_io_group(
    #     model_dir + filename + ".log",
    #     model_dir + filename + ".json",
    #     mode="a"
    # ) if filename else None

    config = dict(
        dataset=dataset,
        scenario=scenario,
        cdm=cdm,
        stream_size=stream_size,
        alpha=alpha,
        beta=beta,
        tolerance=tolerance,
        ctx=ctx,
        epoch=epoch,
        inc_epoch=inc_epoch
    )
    config, reporthook, final_reporthook, tag = prepare_hyper_search(
        config,
        primary_key="ori:accuracy",
        with_keys="id;ori:auc;ori:doa;ori:deltaT;inc:accuracy;inc:auc;inc:doa",
    )
    print(config)
    if wfs is not None:
        print("logs to %s" % model_dir + filename + ".log")
        print(config, file=wfs[0], flush=True)
    dataset_config = {
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            # max_u2i=128,
            # max_i2u=64,
        ),
        "math": dict(
            user_n=10269,
            item_n=17747,
            know_n=1488,
            # max_u2i=128,
            # max_i2u=64,
        ),
    }

    run(
        # cdm="mirt",
        pretrained=pretrained,
        # ctx="cpu",
        reporthook=reporthook,
        final_reporthook=final_reporthook,
        hyper_tag=tag,
        wfs=wfs,
        **config,
        **dataset_config[dataset.split("_")[0]]
    )
    if wfs is not None:
        close_io(wfs)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
