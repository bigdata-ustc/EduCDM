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
from longling.ML.PytorchHelper.toolkit.trainer import collect_params, get_trainer


def run(user_n, item_n, know_n, dataset, scenario, cdm, inc_type=None, stream_size=2048, *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "../../data/%s/" % dataset
    data_dir = dataset_dir + "%s/" % scenario

    cfg = Configuration(
        model_name="%s" % cdm,
        model_dir="%s" % cdm,
        end_epoch=3,
        batch_size=32,
        hyper_params={"user_n": user_n, "item_n": item_n, "know_n": know_n, "cdm": cdm},
        # train_select={".*dtn.*": {}, "^(?!.*dtn)": {'weight_decay': 0}},
        optimizer_params={
            'lr': kwargs.get("lr", 0.002),
            'weight_decay': 0.0001
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

    net = get_net(ctx=cfg.ctx, **cfg.hyper_params)
    print([name for name, _ in net.named_parameters()])
    import re
    select = ".*dtn.*"
    pattern = re.compile(select)
    ret = [name for name, value in net.named_parameters() if pattern.match(name)]
    print(ret)
    select = "^(?!.*dtn)"
    pattern = re.compile(select)
    ret = [name for name, value in net.named_parameters() if pattern.match(name)]
    print(ret)

    print(get_trainer(net, "Adam", optimizer_params=cfg.optimizer_params,
                      select={".*dtn.*": {}, "^(?!.*dtn)": {'weight_decay': 0}}))



if __name__ == '__main__':
    dataset_config = {
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            stream_size=512,
            # max_u2i=64,
            # max_i2u=32
        )
    }
    dataset = "a0910_0.2"
    # scenario = "new_user"
    # scenario = "new_item"
    scenario = "not_distinguish"

    print("+++++++++ %s  +++++++" % scenario)

    run(
        # cdm="mirt",
        cdm="ncd",
        scenario=scenario, dataset=dataset,
        inc_type="global",
        # inc_type="inc",
        ctx="cpu",
        **dataset_config[dataset.split("_")[0]]
    )
