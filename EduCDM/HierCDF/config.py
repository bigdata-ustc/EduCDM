import torch
import os

#Local data path
DATA_PATH = os.path.join(os.getcwd(),'data')

hparams = {
    'n_user': 10000,
    'n_item': 734,
    'n_know': 734,
    'hidden_dim': 1,
    'lr':5e-3, 
    'epoch': 8,
    'batch_size': 4096,
    'logger_mode': 'both', # 'file'/'both'/'console'
    'loss_factor': 0.001,
    'device': torch.device('cpu'), #torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_show': 10,
    'train_ratio': 0.8,
    'itf_type':'irt', # 'irt'/'mirt'/'mf'/'sigmoid-mf'/'ncd'
    'log_path': os.path.join(os.getcwd(),'log'),
    'model_name': os.path.join(os.getcwd(),'model/HierIRT.pkl')
}

