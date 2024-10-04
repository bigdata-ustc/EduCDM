import logging
from EduCDM import CDM, re_index
import pandas as pd
from copy import deepcopy
import torch
from baize.torch import Configuration
from baize.torch import light_module as lm

from EduCDM.ICD.etl import transform, user2items, item2users, dict_etl, Dict2, inc_stream, item2knowledge
from EduCDM.ICD.sym import eval_f, get_net, DualICD, get_dual_loss, dual_fit_f, stableness_eval, turning_point
from EduCDM.ICD.utils import output_metrics


class ICD(CDM):
    r'''
    The IncrementalCD model.

    Args:
        cdm: the base cognitive diagnosis model. Current implemented: 'ncd'
        meta_data: a dictionary containing all the userIds, itemIds, and skills.
        epoch: the training times when a new stream log data comes.
        weight_decay: a optimizer_param reducing the learning rate. Default: 0.1
        inner_metrics: whether to print inner evaluation results on each data stream. Default: True
        logger: whether to log into file. Default: logging
        alpha: a factor balance the accumulated data and incremental data. Default: 0.9
        device: device on which the model is trained. Default: 'cpu'. If you want to run it on your
        GPU, e.g., the first cuda gpu on your machine, you can change it to 'cuda:0'.
    
    Examples:
        meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}

        logger = logging.getLogger("ICD")

        model = ICD('ncd', meta_data, 1, 0.1, True,  logger, 0.9, 'cuda:0')
    '''
    def __init__(self,
                 cdm,
                 meta_data: dict,
                 epoch,
                 weight_decay=0.1,
                 inner_metrics=True,
                 logger=logging,
                 alpha=0.9,
                 device='cpu',
                 **kwargs):
        super(ICD, self).__init__()
        torch.manual_seed(0)
        self.id_reindex, _ = re_index(meta_data)
        user_n = len(self.id_reindex['userId']) + 1
        item_n = len(self.id_reindex['itemId']) + 1
        know_n = len(self.id_reindex['skill']) + 1
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
            optimizer_params={
                'lr': kwargs.get("lr", 0.002),
                'weight_decay': weight_decay
            },
            ctx=device,
            time_digital=True,
        )
        self.logger = logger

        self.net = get_net(**self.cfg.hyper_params, ctx=self.cfg.ctx)
        self.dual_net = DualICD(deepcopy(self.net), self.net, alpha=alpha)
        self.inner_metrics = inner_metrics
        self.u2i, self.i2u, self.i2k = None, None, None

    def transform__(self, df_data: pd.DataFrame):
        items = [self.id_reindex['itemId'][itemId] + 1 for itemId in df_data['itemId'].values]
        skills = []
        for item_skills in df_data['skill']:
            if isinstance(item_skills, str):
                item_skills = eval(item_skills)  # str of list to list
            skills.append([self.id_reindex['skill'][s] + 1 for s in item_skills])
        ret = pd.DataFrame({'itemId': items, 'skill': skills})

        if 'userId' in df_data.columns:
            users = [self.id_reindex['userId'][userId] + 1 for userId in df_data['userId'].values]
            responses = df_data['response'].values
            ret['userId'] = users
            ret['response'] = responses

        return ret

    def fit(self,
            train_data,
            df_item,
            stream_num,
            beta=0.95,
            warmup_ratio=0.1,
            tolerance=1e-3,
            max_u2i=None,
            max_i2u=None,
            hyper_tag=False,
            wfs=None):
        r'''
        Train the model with train_data.

        Args:
            train_data: a dataframe containing training userIds, itemIds and responses.
            df_item: a dataframe containing each item and corresponding skills.
            stream_num: the expected number of streams that the train_data will be devided into.
            beta: a factor balancing the model effectiveness and trait stableness. Default: 0.95
            warmup_ratio: the ratio of train_data that will be used to warm up. The warmup data will be trained in the full training way to implement initialization. Default: 0.1
            tolerance: a factor that determine whether to train the incremental data. Default: 0.001
            max_u2i: the max number of userIds per item. Default: None
            max_i2u: the max number of itemIds per user. Default: None
            hyper_tag: whether to print the verbose information. Default: False
            wfs: whether to save experiment result into file. Default: None
        '''
        dict2 = Dict2()
        act_dual_loss_f = get_dual_loss(self.cfg.ctx, beta=beta)
        warmup_dual_loss_f = get_dual_loss(self.cfg.ctx, beta=1)
        tps = []

        # process data
        train_data = self.transform__(train_data)
        inc_train_df_list = list(inc_stream(train_data, stream_size=int(len(train_data) // stream_num)))
        self.i2k = item2knowledge(self.transform__(df_item))

        warmup = int(warmup_ratio * len(inc_train_df_list))
        train_df = pd.DataFrame()
        for i, inc_train_df in enumerate(inc_train_df_list):
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
                                           self.i2k,
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
                    inc_train_df = train_df = pd.concat([train_df, inc_train_df])
                inc_train_data = transform(inc_train_df,
                                           dict2.u2i,
                                           dict2.i2u,
                                           self.i2k,
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
                    verbose=hyper_tag,
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

            if i + 1 < len(inc_train_df_list) and self.inner_metrics:
                inc_test_data = transform(inc_train_df_list[i + 1],
                                          dict2.u2i,
                                          dict2.i2u,
                                          self.i2k,
                                          self.cfg.hyper_params['know_n'],
                                          max_u2i=max_u2i,
                                          max_i2u=max_i2u,
                                          batch_size=self.cfg.batch_size,
                                          silent=True)
                output_metrics(0, {"tps": tps, "tp_cnt": len(tps), "total": len(inc_train_df_list) - 1},
                               wfs, "tp", self.logger)
                self.eval_prediction(inc_test_data, wfs=wfs)
                if i > 0:
                    self.eval_stableness(pre_dict2, inc_u2i, inc_i2u, wfs=wfs)
        self.u2i = dict2.u2i
        self.i2u = dict2.i2u
    
    def eval_prediction(self, val_data, logger_id=None, wfs=None):
        r'''
        Evaluate the student performance prediction results on the test_data.

        Args:
            val_data: a dataframe containing testing userIds and itemIds.
            logger_id: the id when output the metrics. Default: None
            wfs: whether to save experiment result into file. Default: None
        '''
        inc_met = eval_f(self.net, val_data)
        output_metrics(logger_id, inc_met, wfs, "metrics", self.logger)

    def eval_stableness(self, pre_dict2, inc_u2i, inc_i2u, logger_id=None, wfs=None):
        r'''
        Evaluate the parameter stableness after the final data stream during training.

        Args:
            pre_dict2: a dictionary containing all the userIds, itemIds, and skills of the next incremental data.
            inc_u2i: a dictionary containing all the user data of the incremental data.
            inc_i2u: a dictionary containing all the user data of the incremental data.
            logger_id: the id when output the metrics. Default: None
            wfs: whether to save experiment result into file. Default: None
        '''
        _net = self.dual_net.stat_net
        stat_net = _net.module if isinstance(_net, torch.nn.DataParallel) else _net
        
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
        
        output_metrics(logger_id, sta_met, wfs, "trait", self.logger)
        output_metrics(logger_id, inc_sta_met, wfs, "inc_trait", self.logger)
        
    def predict_proba(self, test_data: pd.DataFrame) -> pd.DataFrame:
        r'''
        Output the predicted probabilities that the users would provide correct answers using test_data.
        The probabilities are within (0, 1).
        
        Args:
            test_data: a dataframe containing testing userIds and itemIds.
        
        Return:
            a dataframe containing the userIds, itemIds, and proba (predicted probabilities).
        '''
        test_data = transform(test_data, self.u2i, self.i2u, self.i2k, self.cfg.hyper_params['know_n'], 32, allow_missing="skip")

        self.net.eval()
        pred_proba = []
        userIds = []
        itemIds = []
        for (uid, u_log, u_mask, iid, i_log, i_mask, i2k, r) in test_data:
            pred, *_ = self.net(u_log, u_mask, i_log, i_mask, i2k)
            pred_proba.extend(pred.tolist())
            userIds.extend(uid.tolist())
            itemIds.extend(iid.tolist())
        ret = pd.DataFrame({'userId': userIds, 'itemId': itemIds, 'proba': pred_proba})
        return ret
    
    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        r'''
        Output the predicted responses using test_data. The responses are either 0 or 1.
    
        Args:
            test_data: a dataframe containing testing userIds and itemIds.
    
        Return:
            a dataframe containing the userIds, itemIds, and predicted responses.
        '''
    
        df_proba = self.predict_proba(test_data)
        y_pred = [1.0 if proba >= 0.5 else 0 for proba in df_proba['proba'].values]
        df_pred = pd.DataFrame({'userId': df_proba['userId'], 'itemId': df_proba['itemId'], 'pred': y_pred})
    
        return df_pred

    def save(self, filepath):
        r'''
        Save the model. This method is implemented based on the PyTorch's torch.save() method. Only the parameters in self.ncdm_net will be saved. You can save the whole NCDM object using pickle.
        
        Args:
            filepath: the path to save the model.
        '''
        
        obj = {'net': self.net.state_dict(), 'dual_net': self.dual_net.state_dict(),
               'u2i': self.u2i, 'i2u': self.i2u, 'i2k': self.i2k, 'cfg': self.cfg}
        torch.save(obj, filepath)
        self.logger.info("save parameters to %s" % filepath)
        
    def load(self, filepath):
        r'''
        Load the model. This method loads the model saved at filepath into self.ncdm_net. Before loading, the object needs to be properly initialized.
        
        Args:
            filepath: the path from which to load the model.
        
        Examples:
            model = KaNCD('ncd', meta_data, epoch=10)  # where meta_data is from the same dataset which is used to train the model at filepath
        
            model.load('path_to_the_pre-trained_model')
        '''
        snapshot = torch.load(filepath, map_location=lambda s, loc: s)
        self.net.load_state_dict(snapshot['net'])
        self.dual_net.load_state_dict(snapshot['dual_net'])
        self.u2i = snapshot['u2i']
        self.i2u = snapshot['i2u']
        self.i2k = snapshot['i2k']
        self.cfg = snapshot['cfg']
        self.logger.info("load parameters from %s" % filepath)
