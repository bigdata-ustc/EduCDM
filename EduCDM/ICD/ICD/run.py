# coding: utf-8

import torch
import pandas as pd
from baize.torch import Configuration
from baize.torch import light_module as lm, load_net
from longling.ML.PytorchHelper import set_device

from ICD.etl import extract, transform, test_etl, merge_dict, dict_etl
from sym import fit_f, eval_f, get_loss, get_net, stableness_eval


def run(cdm, user_n, item_n, know_n, dataset, scenario, max_u2i=None, max_i2u=None, pretrained=False,
        inc_type="inc", *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "../../data/%s/" % dataset
    data_dir = dataset_dir + "%s/" % scenario

    cfg = Configuration(
        model_name="icd_%s" % cdm,
        model_dir="icd_%s" % cdm,
        end_epoch=1,
        batch_size=32,
        hyper_params={"user_n": user_n, "item_n": item_n, "know_n": know_n, "cdm": cdm},
        # train_select={
            # ".*dtn.*": {}, "^(?!.*dtn)": {'weight_decay': 0}
        # },
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            # 'weight_decay': 0.0001
        },
        ctx=kwargs.get("ctx", "cuda: 0")
    )
    print(cfg)

    item2know = "%s/item.csv" % dataset_dir
    path_format = "%s/{}.csv" % data_dir

    stat_train_data_path = path_format.format("stat_train")
    stat_valid_data_path = path_format.format("stat_valid")
    stat_test_data_path = path_format.format("stat_test")

    inc_train_data_path = path_format.format("inc_train")
    inc_valid_data_path = path_format.format("inc_valid")
    inc_test_data_path = path_format.format("inc_test")

    train_df, u2i, i2u, i2k = extract(stat_train_data_path, item2know)

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
        print(eval_f(net, stat_test_data))

        if cdm == "ncd":
            print("finetune int fc")
            # cfg.train_select = ".*int.*"
            # cfg.optimizer_params["lr"] = 0.0001
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
            print("Ori.")
            print(eval_f(net, stat_test_data))
    else:
        net = set_device(net, cfg.ctx)

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

    inc_train_df, inc_u2i, inc_i2u, _ = extract(inc_train_data_path, item2know)
    merge_dict(u2i, inc_u2i)
    merge_dict(i2u, inc_i2u)

    inc_valid_data = list(test_etl(inc_valid_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))
    inc_test_data = list(test_etl(inc_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))

    print("=============== Stat. ===================")

    print("Ori.")
    print(eval_f(net, stat_test_data))

    print("Inc.")
    print(eval_f(net, inc_test_data))

    stat_net = net.module if isinstance(net, torch.nn.DataParallel) else net

    users = list(u2i.keys())
    items = list(i2u.keys())
    user_traits = stat_net.get_user_profiles(dict_etl(users, u2i, batch_size=cfg.batch_size))
    item_traits = stat_net.get_item_profiles(dict_etl(items, i2u, batch_size=cfg.batch_size))

    if inc_type == "global":
        inc_train_df = pd.concat([train_df, inc_train_df])

    inc_train_data = transform(
        inc_train_df, u2i, i2u, i2k, know_n, cfg.batch_size,
        max_u2i=max_u2i, max_i2u=max_i2u, silent=True
    )

    # cfg.train_select = "^(?!.*dtn)"
    print(cfg)

    lm.train(
        net=net,
        cfg=cfg,
        loss_function=loss_f,
        trainer=None,
        train_data=inc_train_data,
        test_data=inc_valid_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=False,
    )

    stat_test_data = list(test_etl(stat_test_data_path, u2i, i2u, i2k, know_n, cfg.batch_size))

    print("=============== %s ===================" % inc_type)

    print("Ori.")
    print(eval_f(net, stat_test_data))

    print("Inc.")
    print(eval_f(net, inc_test_data))

    print("Trait")
    print(stableness_eval(net, users, items, u2i, i2u, user_traits, item_traits, cfg.batch_size))


if __name__ == '__main__':
    dataset_config = {
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            # max_u2i=64,
            # max_i2u=32
        ),
        "xunfei": dict(
            user_n=10268 + 1,
            item_n=2507 + 1,
            know_n=497,
            # max_u2i=64,
            # max_i2u=32
        )
    }
    dataset = "a0910"
    scenario = "new_item"

    run(
        cdm="ncd",
        # cdm="mirt",
        # pretrained=True,
        scenario=scenario,
        dataset=dataset,
        ctx="cuda: 0",
        # ctx="cpu",
        **dataset_config[dataset]
    )
