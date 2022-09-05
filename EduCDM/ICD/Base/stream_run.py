# coding: utf-8
import torch
import pandas as pd
from baize.torch import Configuration
from baize.torch import light_module as lm
from baize.torch import save_params
from longling import build_dir

from ICD.etl import inc_stream
from etl import extract, transform, etl, item2knowledge
from sym import fit_f, eval_f, get_loss, get_net, stableness_eval


def run(user_n, item_n, know_n, dataset, scenario, cdm, inc_type=None, stream_size=2048, *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "../../data/%s/" % dataset
    data_dir = dataset_dir + "%s/" % scenario

    cfg = Configuration(
        model_name="%s" % cdm,
        model_dir="%s" % cdm,
        end_epoch=3,
        batch_size=32,
        hyper_params={"user_num": user_n, "item_num": item_n, "know_n": know_n},
        # train_select={".*int.*": {'weight_decay': 0}, "^(?!.*int)": {}},
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            # 'weight_decay': 1e-6
        },
        ctx=kwargs.get("ctx", "cuda: 3")
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

    i2k = item2knowledge(item2know)

    train_df = extract(stat_train_data_path)
    stat_train_data = transform(train_df, i2k, know_n, cfg.batch_size)

    stat_valid_data = etl(stat_valid_data_path, i2k, know_n, cfg.batch_size)
    stat_test_data = etl(stat_test_data_path, i2k, know_n, cfg.batch_size)

    net = get_net(ctx=cfg.ctx, cdm=cdm, **cfg.hyper_params)
    loss_f = get_loss(ctx=cfg.ctx)

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

    print("===================== Stat. ======================")
    print("Ori.")
    print(eval_f(net, stat_test_data))

    save_net = net.module if isinstance(net, torch.nn.DataParallel) else net
    user = list(set(train_df["user_id"].tolist()))
    item = list(set(train_df["item_id"].tolist()))

    user_traits = save_net.get_user_profiles(user)
    item_traits = save_net.get_item_profiles(item)

    if inc_type is None:
        build_dir(data_dir + "/model/%s/" % cdm)
        save_net.save_user_profiles(user, data_dir + "/model/%s/user.pt" % cdm)
        save_net.save_item_profiles(item, data_dir + "/model/%s/item.pt" % cdm)
        if cdm == "ncd":
            save_net.save_cdm(data_dir + "/model/%s/cdm.params" % cdm)
        return

    # stream inc
    inc_train_df = extract(inc_train_data_path)
    inc_train_df_list = list(inc_stream(inc_train_df, stream_size=stream_size))
    inc_valid_data = etl(inc_valid_data_path, i2k, know_n, cfg.batch_size)
    inc_test_data = etl(inc_test_data_path, i2k, know_n, cfg.batch_size)

    cfg.train_select = "^(?!.*int_fc)"
    print(cfg)
    for i, inc_train_df in enumerate(inc_train_df_list):
        print("============= Stream[%s/%s] =============" % (i, len(inc_train_df_list)))
        if inc_type == "global":
            train_df = pd.concat([train_df, inc_train_df])
            inc_train_df = train_df

        inc_train_data = transform(inc_train_df, i2k, know_n, cfg.batch_size)

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
        if i % max(round(len(inc_train_df_list) // 10), 1) == 0:
            print("===================== %s valid ======================" % inc_type)
            print("Ori.")
            print(eval_f(net, stat_valid_data))
            print("Inc.")
            print(eval_f(net, inc_valid_data))
            print("Trait")
            print(stableness_eval(net, user, item, user_traits, item_traits))

    print("===================== %s ======================" % inc_type)
    print("Ori.")
    print(eval_f(net, stat_test_data))
    print("Inc.")
    print(eval_f(net, inc_test_data))
    print("Trait")
    print(stableness_eval(net, user, item, user_traits, item_traits))


if __name__ == '__main__':
    dataset_config = {
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            # max_u2i=64,
            # max_i2u=32
        ),
        "math": dict(
            user_n=10269,
            item_n=17747,
            know_n=1488,
            # max_u2i=128,
            # max_i2u=256,
        ),
    }
    dataset = "a0910_small"
    # scenario = "new_user"
    scenario = "new_item"
    # scenario = "not_distinguish"

    print("+++++++++ %s  +++++++" % scenario)

    run(
        # cdm="mirt",
        cdm="ncd",
        scenario=scenario, dataset=dataset,
        inc_type="global",
        stream_size=2048,
        # inc_type="inc",
        ctx="cuda: 2",
        **dataset_config[dataset.split("_")[0]]
    )
