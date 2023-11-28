import numpy as np
import pandas as pd
import torch

class TrainDataLoader:
    def __init__(self, train_data: pd.DataFrame, Q_matrix: np.array, batch_size: int):
        self.train_data = train_data
        self.Q_matrix = Q_matrix
        self.batch_size = batch_size
        self.n_sample = train_data.shape[0]
        self.cursor = 0
    
    def next_batch(self)->list:
        if self.is_end():
            return None, None, None
        stu_ids, exer_ids, y_labels = [], [], []
        for cptr in range(self.batch_size):
            if self.cursor + cptr >= self.n_sample:
                break
            record = self.train_data.iloc[self.cursor + cptr,:]
            stu_ids.append(record['user_id'])
            exer_ids.append(record['exer_id'])
            y_labels.append(record['score'])

        self.cursor += self.batch_size
        item_know = self.Q_matrix[exer_ids,:]

        return torch.LongTensor(stu_ids), torch.LongTensor(exer_ids), torch.DoubleTensor(item_know), torch.LongTensor(y_labels)
    
    def is_end(self)->bool:
        return self.cursor >= self.n_sample

    def reset(self):
        self.cursor = 0
        return

if __name__ == "__main__":
    tdl = TrainDataLoader()
    print(tdl.data)