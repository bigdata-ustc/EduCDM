import logging
from EduCDM import CDM
import pandas as pd
from copy import deepcopy
import torch
from baize.torch import Configuration
from baize.torch import light_module as lm

from EduCDM.ICD.etl import transform, user2items, item2users, dict_etl, Dict2
from EduCDM.ICD.sym import eval_f, get_net, DualICD, get_dual_loss, dual_fit_f, stableness_eval, turning_point
from EduCDM.ICD.utils import output_metrics


class ICD(CDM):
    def __init__(self,
                 cdm,
                 user_n,
                 item_n,
                 know_n,
                 epoch=1,
                 weight_decay=0.1,
                 inner_metrics=True,
                 logger=logging,
                 alpha=0.9,
                 ctx='cpu',
                 **kwargs):
        super(ICD, self).__init__()
        torch.manual_seed(0)
        self.cfg = Configuration(
            model_name="icd_%s" % cdm,
            model_dir="icd_%s" % cdm,
            end_epoch=epoch,
            batch_size=32,
            hyper_params={
                "user_n": user_n,
                "item_n": item_n,
                "know_n": know_n,
                "cdm": cdm
            },
            # train_select={".*dtn.*": {}, "^(?!.*dtn)": {'weight_decay': 0}},
            optimizer_params={
                'lr': kwargs.get("lr", 0.002),
                'weight_decay': weight_decay
            },
            ctx=ctx,
            time_digital=True,
        )
        self.logger = logger

        self.net = get_net(**self.cfg.hyper_params, ctx=self.cfg.ctx)
        self.dual_net = DualICD(deepcopy(self.net), self.net, alpha=alpha)
        self.inner_metrics = inner_metrics

    def train(self,
              inc_train_df_list,
              i2k,
              beta=0.95,
              warmup_ratio=0.1,
              tolerance=1e-3,
              max_u2i=None,
              max_i2u=None,
              hyper_tag=False,
              vector_numbers=None,
              vector_path_format=None,
              wfs=None):
        vector_user = None
        vector_item = None
        dict2 = Dict2()
        act_dual_loss_f = get_dual_loss(self.cfg.ctx, beta=beta)
        warmup_dual_loss_f = get_dual_loss(self.cfg.ctx, beta=1)
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
            self.dual_net.stat_net = deepcopy(self.dual_net.net)

            self.logger.info("============= Stream[%s/%s/%s] =============" %
                             (i, len(tps), len(inc_train_df_list)))
            pre_net = deepcopy(self.net)
            pre_net.eval()

            if i < warmup or turning_point(self.net,
                                           inc_train_df,
                                           dict2,
                                           inc_dict2,
                                           i2k,
                                           self.cfg.hyper_params['know_n'],
                                           self.cfg.batch_size,
                                           ctx=self.cfg.ctx,
                                           tolerance=tolerance,
                                           logger=self.logger):
                self.logger.info("**** Turning Point ****")
                tps.append(i)

                dict2.merge_u2i(inc_u2i)
                dict2.merge_i2u(inc_i2u)
                if i < warmup:
                    inc_train_df = train_df = pd.concat(
                        [train_df, inc_train_df])
                inc_train_data = transform(inc_train_df,
                                           dict2.u2i,
                                           dict2.i2u,
                                           i2k,
                                           self.cfg.hyper_params['know_n'],
                                           max_u2i=max_u2i,
                                           max_i2u=max_i2u,
                                           batch_size=self.cfg.batch_size,
                                           silent=True)
                lm.train(
                    net=self.dual_net,
                    cfg=self.cfg,
                    loss_function=dual_loss_f,
                    trainer=None,
                    train_data=inc_train_data,
                    fit_f=dual_fit_f,
                    eval_f=eval_f,
                    initial_net=False,
                    verbose=not hyper_tag,
                )
                if i > warmup:
                    self.dual_net.momentum_weight_update(
                        pre_net, self.cfg.train_select)
                self.dual_net.stat_net = pre_net

            else:
                dict2.merge_u2i(inc_u2i)
                dict2.merge_i2u(inc_i2u)

            dict2.merge_u2i_r(inc_dict2)
            dict2.merge_i2u_r(inc_dict2)

            if i == 0:
                vector_user = list(dict2.u2i.keys())
                vector_item = list(dict2.i2u.keys())

            if vector_numbers and i in vector_numbers:  # pragma: no cover
                _net = self.dual_net.net
                trait_net = _net.module if isinstance(
                    _net, torch.nn.DataParallel) else _net
                vector_user_traits = trait_net.get_user_profiles(
                    dict_etl(vector_user,
                             pre_dict2.u2i,
                             batch_size=self.cfg.batch_size))
                vector_item_traits = trait_net.get_item_profiles(
                    dict_etl(vector_item,
                             pre_dict2.i2u,
                             batch_size=self.cfg.batch_size))
                vector_path = vector_path_format.format("user", i)
                self.logger.info("user traits to %s" % vector_path)
                torch.save(vector_user_traits, vector_path)
                vector_path = vector_path_format.format("item", i)
                self.logger.info("item traits to %s" % vector_path)
                torch.save(vector_item_traits, vector_path)
            if i + 2 == len(inc_train_df_list) or self.inner_metrics:
                inc_test_data = transform(inc_train_df_list[i + 1],
                                          dict2.u2i,
                                          dict2.i2u,
                                          i2k,
                                          self.cfg.hyper_params['know_n'],
                                          max_u2i=max_u2i,
                                          max_i2u=max_i2u,
                                          batch_size=self.cfg.batch_size,
                                          silent=True)
                self.eval(i, inc_train_df_list, inc_test_data, pre_dict2,
                          inc_u2i, inc_i2u, tps, wfs)

    def eval(self, i, inc_train_df_list, inc_test_data, pre_dict2, inc_u2i,
             inc_i2u, tps, wfs):
        inc_met = eval_f(self.net, inc_test_data)
        output_metrics(i, inc_met, wfs, "metrics", self.logger)
        if i > 0:
            _net = self.dual_net.stat_net
            stat_net = _net.module if isinstance(
                _net, torch.nn.DataParallel) else _net

            users = list(pre_dict2.u2i.keys())
            items = list(pre_dict2.i2u.keys())
            user_traits = stat_net.get_user_profiles(
                dict_etl(users, pre_dict2.u2i, batch_size=self.cfg.batch_size))
            item_traits = stat_net.get_item_profiles(
                dict_etl(items, pre_dict2.i2u, batch_size=self.cfg.batch_size))
            sta_met = stableness_eval(self.dual_net.net, users, items,
                                      pre_dict2.u2i, pre_dict2.i2u,
                                      user_traits, item_traits,
                                      self.cfg.batch_size)

            inc_users = list(inc_u2i.keys())
            inc_items = list(inc_i2u.keys())
            inc_user_traits = stat_net.get_user_profiles(
                dict_etl(inc_users, inc_u2i, batch_size=self.cfg.batch_size))
            inc_item_traits = stat_net.get_item_profiles(
                dict_etl(inc_items, inc_i2u, batch_size=self.cfg.batch_size))
            inc_sta_met = stableness_eval(self.dual_net.net, inc_users,
                                          inc_items, inc_u2i, inc_i2u,
                                          inc_user_traits, inc_item_traits,
                                          self.cfg.batch_size)

            output_metrics(i, sta_met, wfs, "trait", self.logger)
            output_metrics(i, inc_sta_met, wfs, "inc_trait", self.logger)

        output_metrics(0, {
            "tps": tps,
            "tp_cnt": len(tps),
            "total": len(inc_train_df_list) - 1
        }, wfs, "tp", self.logger)

    def save(self):
        pass

    def load(self):
        pass
