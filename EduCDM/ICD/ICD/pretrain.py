# coding: utf-8

import torch

from sym import ICD
from baize.utils import pad_sequence
from baize.torch import Configuration, load_net, light_module as lm, fit_wrapper, loss_dict2tmt_torch_loss, save_params
from tqdm import tqdm
from ICD.etl import extract, test_etl
from sym import eval_f


def user_etl(user_obj, u2i, batch_size):
    def pack_batch(_batch):
        __u2i, __u_trait = zip(*_batch)
        user_items_length = [len(d) for d in __u2i]
        return (
            torch.tensor(pad_sequence(__u2i), dtype=torch.int64),
            torch.tensor(user_items_length, dtype=torch.int64),
            torch.tensor(__u_trait),
        )

    batch_data = []
    batch = []
    uid = user_obj["uid"].tolist()
    u_trait = user_obj["u_trait"].tolist()
    for idx, _uid in tqdm(enumerate(uid), "user batchify"):
        _u_trait = u_trait[idx]
        batch.append((u2i[_uid], _u_trait))
        if len(batch) == batch_size:
            batch_data.append(pack_batch(batch))
            batch = []
    if batch:
        batch_data.append(pack_batch(batch))

    return batch_data


@fit_wrapper
def user_fit_f(_net, batch_data, loss_function, *args, **kwargs):
    u2i, u2i_mask, u_trait = batch_data
    u_trait = _net.l_dtn(u2i, u2i_mask)
    out = _net.cdm.u_theta(u_trait)
    loss = []
    for _f in loss_function.values():
        loss.append(_f(out, u_trait))
    return sum(loss)


def item_etl(item_obj, i2u, batch_size):
    def pack_batch(_batch):
        __i2u, __i_kd, __i_ed = zip(*_batch)
        item_users_length = [len(d) for d in __i2u]
        return (
            torch.tensor(pad_sequence(__i2u), dtype=torch.int64),
            torch.tensor(item_users_length, dtype=torch.int64),
            torch.tensor(__i_kd),
            torch.tensor(__i_ed),
        )

    batch_data = []
    batch = []
    iid = item_obj["iid"].tolist()
    i_kd = item_obj["ib"].tolist()
    i_ed = item_obj["ia"].tolist()
    for idx, _iid in tqdm(enumerate(iid), "item batchify"):
        _i_kd = i_kd[idx]
        _i_ed = i_ed[idx]
        batch.append((i2u[_iid], _i_kd, _i_ed))
        if len(batch) == batch_size:
            batch_data.append(pack_batch(batch))
            batch = []
    if batch:
        batch_data.append(pack_batch(batch))

    return batch_data


@fit_wrapper
def item_fit_f(_net, batch_data, loss_function, *args, **kwargs):
    i2u, i2u_mask, i_kd, i_ed = batch_data
    i_trait = _net.i_dtn(i2u, i2u_mask)
    out_kd = _net.cdm.i_difficulty(i_trait)
    out_ed = _net.cdm.i_discrimination(i_trait)
    loss = []
    for _f in loss_function.values():
        loss.append(_f(out_kd, i_kd) + _f(out_ed, i_ed))
    return sum(loss)


def run(cdm, user_n, item_n, know_n, dataset, scenario, max_u2i=None, max_i2u=None, *args, **kwargs):
    torch.manual_seed(0)

    dataset_dir = "../../data/%s/" % dataset
    data_dir = dataset_dir + "%s/" % scenario
    model_dir = data_dir + "model/"

    cfg = Configuration(
        model_name="icd_%s" % cdm,
        model_dir="icd_%s" % cdm,
        end_epoch=5,
        batch_size=32,
        hyper_params={"user_n": user_n, "item_n": item_n, "know_n": know_n, "cdm": cdm},
        optimizer_params={'lr': kwargs.get("lr", 0.002), 'weight_decay': 0.0001},
        ctx=kwargs.get("ctx", "cuda: 0")
    )

    _, u2i, i2u, i2k = extract(data_dir + "stat_train.csv", dataset_dir + "item.csv")

    user_profiles = model_dir + "%s/user.pt" % cdm
    item_profiles = model_dir + "%s/item.pt" % cdm

    icd = ICD(**cfg.hyper_params)
    if cdm == "ncd":
        cdm_params = model_dir + "%s/cdm.params" % cdm
        load_net(cdm_params, icd.cdm.int_fc)

    loss_f = loss_dict2tmt_torch_loss({"l2": torch.nn.MSELoss()})

    user_obj = torch.load(user_profiles)
    user_train_data = user_etl(user_obj, u2i, cfg.batch_size)

    cfg.train_select = ".*l_dtn.*"
    # cfg.optimizer_params["lr"] = 0.001
    print(cfg)
    lm.train(
        net=icd,
        cfg=cfg,
        loss_function=loss_f,
        trainer=None,
        train_data=user_train_data,
        fit_f=user_fit_f,
        initial_net=False,
    )

    item_obj = torch.load(item_profiles)
    item_train_data = item_etl(item_obj, i2u, cfg.batch_size)

    cfg.train_select = ".*i_dtn.*"
    # cfg.optimizer_params["lr"] = 0.001
    print(cfg)
    lm.train(
        net=icd,
        cfg=cfg,
        loss_function=loss_f,
        trainer=None,
        train_data=item_train_data,
        fit_f=item_fit_f,
        initial_net=False,
    )

    stat_test_data = list(test_etl(data_dir + "stat_test.csv", u2i, i2u, i2k, know_n, cfg.batch_size))
    print(eval_f(icd, stat_test_data))

    params_path = model_dir + "%s/icd_%s_init.params" % (cdm, cdm)
    print("save model to %s" % params_path)
    save_params(params_path, icd)


if __name__ == '__main__':
    dataset_config = {
        "a0910": dict(
            user_n=4129,
            item_n=17747,
            know_n=123,
            # max_u2i=64,
            # max_i2u=32
        )
    }
    dataset = "a0910"
    scenario = "new_item"

    run(cdm="ncd", scenario=scenario, dataset=dataset, lr=0.002, ctx="cuda: 0", **dataset_config[dataset])
