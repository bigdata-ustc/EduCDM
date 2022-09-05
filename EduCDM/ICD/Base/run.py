# coding: utf-8
import torch
import pandas as pd
import random
from baize.torch import Configuration
from baize.torch import light_module as lm
from longling import build_dir

from etl import extract, transform, etl, item2knowledge
from sym import fit_f, eval_f, get_loss, get_net, stableness_eval


def run(user_n, item_n, know_n, dataset, scenario, cdm, inc_type=None, *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "../../data/%s/" % dataset
    dataset_dir = "/home/yutingh/icd/data/%s/" % dataset

    data_dir = dataset_dir + "%s/" % scenario

    cfg = Configuration(
        model_name="%s" % cdm,
        model_dir="%s" % cdm,
        end_epoch=4,
        batch_size=32,
        hyper_params={"user_num": user_n,
                      "item_num": item_n, "know_n": know_n},
        # train_select={".*int_fc.*": {'weight_decay': 0}, "^(?!.*int_fc)": {}},
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            # 'weight_decay': 0.0001,
            'weight_decay': 1e-6,
        },
        ctx=kwargs.get("ctx", "cuda: 3"),
        time_digital=True,
        rand_num=str(int(random.random()*10000))
    )
    print(scenario, '\n', inc_type)
    print(cfg)
    csv_path = "/home/yutingh/icd/ICD/Base/" + cdm+"_" + scenario+"_"+inc_type+'_lr' + \
        str(cfg.optimizer_params['lr'])+'_epoch' + \
        str(cfg.end_epoch)+"_"+cfg.rand_num+".csv"

    item2know = "%sitem.csv" % dataset_dir
    path_format = "%s{}.csv" % data_dir

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
    save_report_to_csv(eval_f(net, stat_test_data), csv_path)

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

    inc_train_df = extract(inc_train_data_path)
    inc_valid_data = etl(inc_valid_data_path, i2k, know_n, cfg.batch_size)
    inc_test_data = etl(inc_test_data_path, i2k, know_n, cfg.batch_size)

    if inc_type == "global":
        inc_train_data = transform(
            pd.concat([train_df, inc_train_df]), i2k, know_n, cfg.batch_size)
    else:
        inc_train_data = transform(inc_train_df, i2k, know_n, cfg.batch_size)

    cfg.train_select = "^(?!.*int_fc.*)"
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
    print("===================== %s ======================" % inc_type)
    print("Ori.")
    save_report_to_csv(eval_f(net, stat_test_data), csv_path)
    print("Inc.")
    save_report_to_csv(eval_f(net, inc_test_data), csv_path)
    print("Trait")
    df = pd.DataFrame(stableness_eval(net, user, item,
                                      user_traits, item_traits)).transpose()
    print(df)
    df = df.round(3)
    df1 = df.loc[['macro ave', 'micro ave']].transpose()
    df = df.reindex(['theta', 'a', 'b'])
    df1 = df1.reindex(['delta'])
    df1.index = pd.Series([' '])
    df1 = df1.append(pd.Series({'macro ave': ''}, name=" "))
    df.to_csv(csv_path, mode='a', index=True)
    df1.to_csv(csv_path, mode='a', index=True)


def save_report_to_csv(df, csv_path):
    df = pd.DataFrame(df).transpose()
    df = df.round(3)
    df1 = df.loc[['accuracy', 'macro_auc', 'doa', 'macro_aupoc',
                  'doa_know_support', 'doa_z_support']].transpose()
    df = df.reindex(['0.0', '1.0', 'macro_avg'])
    # df1 = df1[['accuracy', 'macro_auc', 'macro_aupoc',
    #            'doa', 'doa_know_support', 'doa_z_support']]
    df1 = df1.reindex(['precision'])
    df1.index = pd.Series([' '])
    df1 = df1.append(pd.Series({'accuracy': ''}, name=" "))
    df.to_csv(csv_path, mode='a', index=True)
    df1.to_csv(csv_path, mode='a', index=True)


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
    dataset = "a0910"
    # scenario = "new_user"
    # scenario = "new_item"
    scenario = "not_distinguish"

    print("+++++++++ %s  +++++++" % scenario)
    run(
        # cdm="mirt",
        cdm="ncd",
        scenario=scenario, dataset=dataset,
        inc_type="global",
        lr=0.01,
        # ctx="cpu",
        inc_type="inc", ctx="cuda:0",
        **dataset_config[dataset]
    )
