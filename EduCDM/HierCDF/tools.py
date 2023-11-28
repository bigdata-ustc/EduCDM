import os
import numpy as np
import pandas as pd
import time
import torch
from datetime import datetime

class Logger:
    '''
    logger suitable for anywhere
    '''
    def __init__(self, path = '/home/lijt/HCD/HCD_Workspace/HierCDM/log/', mode='both'):
        self.fmt = "%Y-%m-%d-%H:%M:%S"
        self.begin_time = time.strftime(self.fmt,time.localtime())
        self.path = os.path.join(path,self.begin_time +'/')

    
    def write(self, message: str, mode = 'both'):
        '''
        @Param mode: 
            file(default): print to file
            console: print to screen
            both: print to both file and screen
        '''
        current_time = time.strftime(self.fmt,time.localtime())
        begin = datetime.strptime(self.begin_time,self.fmt)
        end = datetime.strptime(current_time, self.fmt)
        minutes = (end - begin).seconds
        record = '{} ({} s used) {}\n'.format(current_time , minutes, message)

        if mode == 'file' or mode == 'both':
            if not os.path.exists(self.path):
                os.makedirs(self.path)

        if mode == 'file':
            with open(self.path+'log.txt','a') as f:
                f.write(record)

        elif mode == 'console':
            print(record, end='')
        
        elif mode == 'both':
            with open(self.path+'log.txt','a') as f:
                f.write(record)
            print(record, end='')
        
        else:
            print('Logger error! [mode] must be \'file\' or \'console\' or \'both\'.')

def labelize(y_pred: torch.DoubleTensor, threshold = 0.5)->np.ndarray:
    return (y_pred > threshold).to('cpu').detach().numpy().astype(np.int).reshape(-1,)

def to_numpy(y_pred: torch.DoubleTensor)->np.ndarray:
    return y_pred.to('cpu').detach().numpy().reshape(-1,)

def df_preview(df: pd.DataFrame):
    print('columns =',df.columns)
    print('shape =',df.shape)
    print(df.head())
    describe = df.describe()
    unique_dict = {}
    for col in describe.columns:
        unique_dict[col]=np.unique(df.loc[:,col]).shape[0]
    describe.loc['unique'] = unique_dict
    print(describe)

def divide_train_test(data: pd.DataFrame, train_ratio: float = 0.5)->list:
    train_data = pd.DataFrame(columns = data.columns)
    test_data = pd.DataFrame(columns = data.columns)
    
    group_user = data.groupby('user_id')
    for user_id, group in group_user:
        user_log = group.reset_index(drop=True)
        n_user_log = group.shape[0]
        n_train = int(n_user_log * train_ratio)
        train_data = train_data.append(user_log.iloc[:n_train,:], ignore_index = True)
        test_data = test_data.append(user_log.iloc[n_train:,:], ignore_index = True)

    return train_data, test_data