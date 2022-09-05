# coding: utf-8
from torch import nn
from tqdm import tqdm
from baize.metrics import classification_report, POrderedDict
from baize.torch import fit_wrapper, eval_wrapper
from ICD.metrics import doa_report, stableness_report


@fit_wrapper
def fit_f(_net, batch_data, loss_function, *args, **kwargs):
    uid, iid, i2k, r = batch_data
    out = _net(uid, iid, i2k)

    loss = []
    for _f in loss_function.values():
        loss.append(_f(out, r))
    return sum(loss)


@eval_wrapper
def eval_f(_net, test_data, *args, **kwargs):
    y_true = []
    y_pred = []
    y_label = []
    user_id = []
    item_id = []
    user_theta = []
    item_knowledge = []

    theta_net = _net.module if isinstance(_net, nn.DataParallel) else _net
    device = next(theta_net.parameters()).device
    for uid, iid, i2k, r in tqdm(test_data, "evaluating"):
        pred = _net(uid, iid, i2k)
        y_pred.extend(pred.tolist())
        y_label.extend([0 if p < 0.5 else 1 for p in pred])
        y_true.extend(r.tolist())

        user_id.extend(uid.tolist())
        item_id.extend(iid.tolist())
        user_theta.extend(theta_net.u_theta(uid.to(device)).tolist())
        item_knowledge.extend(i2k.tolist())

    try:
        ret = classification_report(y_true, y_label, y_pred)
    except ValueError:
        ret = POrderedDict()
    ret.update(doa_report(user_id, item_id, item_knowledge, y_true, user_theta))
    return ret


@eval_wrapper
def stableness_eval(net, user, item, user_traits, item_traits):
    new_net = net.module if isinstance(net, nn.DataParallel) else net
    new_user_traits = new_net.get_user_profiles(user)
    new_item_traits = new_net.get_item_profiles(item)

    return stableness_report(
        [user_traits["u_trait"], item_traits['ia'], item_traits['ib']],
        [new_user_traits["u_trait"], new_item_traits['ia'], new_item_traits['ib']],
        ['theta', 'a', 'b']
    )
