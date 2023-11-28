import gc
import os
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
import sys

sys.path.append(os.getcwd())

from tools import Logger, labelize, to_numpy, divide_train_test
from config import DATA_PATH, hparams
from model import HierCDF

def train_test(data: pd.DataFrame, valid_data: pd.DataFrame = None, know_graph: pd.DataFrame = None, Q_matrix: np.array = None) -> dict:
    n_data = data.shape[0]
    n_user = hparams['n_user'] 
    n_item = hparams['n_item']
    n_know = hparams['n_know']
    hidden_dim = hparams['hidden_dim']
    device = hparams['device']
    logger_mode = hparams['logger_mode']
    train_ratio = hparams['train_ratio']
    log_path = hparams['log_path']
    itf_type = hparams['itf_type']
    model_name = hparams['model_name']

    model = HierCDF(n_user, n_item, n_know, hidden_dim, know_graph,itf_type, log_path)
    if valid_data is None:
        train_data, valid_data = divide_train_test(data, train_ratio)
    else:
        train_data = data
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    model.logger.write("{} {}".format(train_data.shape, valid_data.shape),logger_mode)
    model.train(hparams = hparams, train_data = train_data, valid_data = valid_data, Q_matrix = Q_matrix)
    model.save(model_name)

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    #data = pd.read_csv(osp.join(DATA_PATH,'data.csv'), index_col=0)
    know_graph = pd.read_csv(osp.join(DATA_PATH,'hier.csv'))
    #know_graph=pd.DataFrame()
    Q_matrix = np.loadtxt(osp.join(DATA_PATH, 'Q_matrix.txt'), delimiter=' ')
    n_user=hparams['n_user']
    n_item=hparams['n_item']
    n_know=hparams['n_know']
    train_data=pd.read_csv(osp.join(DATA_PATH,'train_0.8_0.2.csv'))
    test_data=pd.read_csv(osp.join(DATA_PATH,'test_0.8_0.2.csv'))
    for i in range(1):
        train_test(train_data, test_data, know_graph, Q_matrix)

