# HierCDF

The implementation of the HierCDF model in paper: [HierCDF: A Bayesian Network-based Hierarchical Cognitive Diagnosis Framework](https://dl.acm.org/doi/10.1145/3534678.3539486)



If this code helps with your studies, please kindly cite the following publication:

```
@inproceedings{10.1145/3534678.3539486,
author = {Li, Jiatong and Wang, Fei and Liu, Qi and Zhu, Mengxiao and Huang, Wei and Huang, Zhenya and Chen, Enhong and Su, Yu and Wang, Shijin},
title = {HierCDF: A Bayesian Network-Based Hierarchical Cognitive Diagnosis Framework},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3534678.3539486},
doi = {10.1145/3534678.3539486},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {904–913},
numpages = {10},
location = {Washington DC, USA},
series = {KDD '22}
}
```


## Parameters description

| Parameters | Type | Description                              |
| ---------- | ---- | ---------------------------------------- |
| meta_data  | dict | a dictionary containing all the userIds, itemIds, and skills. |
| knowledge_graph        | pandas.DataFrame  | the data frame that contains the knowledge graph, whose columns = ['source', 'target']. Each row represents an edge from the 'source' vertex to the 'target' vertex|
| hidd_dim    | int  | the dimension of inner layers of HierCDF |



### Examples

```python
import numpy as np
import pandas as pd
from EduCDM import HierCDF

if __name__ == '__main__':

    # Generate dataset
    train_data = pd.DataFrame({
        'userId': [
            '001', '001', '001', '001', '002', '002',
            '002', '002', '003', '003', '003', '003'],
        'itemId': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'response': [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1]
    })
    know_graph = pd.DataFrame({
        'source': [0, 0, 1],
        'target': [1, 2, 3]
    })
    q_matrix = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1]
    ])

    # Initialize the 'skill' column of train_data
    train_data['skill'] = 0
    for id in range(train_data.shape[0]):
        item_id = train_data.loc[id, 'itemId']
        concepts = np.where(
            q_matrix[item_id] > 0)[0].tolist()
        train_data.loc[id, 'skill'] = str(concepts)
    
    # Generate meta_data
    meta_data = {'userId': [], 'itemId': [], 'skill': []}
    meta_data['userId'] = train_data['userId'].unique().tolist()
    meta_data['itemId'] = train_data['itemId'].unique().tolist()
    meta_data['skill'] = [i for i in range(q_matrix.shape[1])]

    hiercdm = HierCDF(meta_data, know_graph, hidd_dim=32)
    hiercdm.fit(
        train_data,
        val_data=train_data,
        batch_size=1, epoch=3, lr=0.01)
    hiercdm.save('./hiercdf.pt')
    new_hiercdm = HierCDF(meta_data, know_graph, hidd_dim=32)
    new_hiercdm.load('./hiercdf.pt')
    new_hiercdm.fit(
        train_data,
        val_data=train_data,
        batch_size=1, epoch=1, lr=0.01)
    new_hiercdm.eval(train_data)
```



## Methods summary

| Methods           | Description                              |
| ----------------- | ---------------------------------------- |
| fit               | Fits the model to the training data.     |
| fit_predict       | Use the model to predict the responses in the testing data and returns the results. The responses are either 1 (i.e., correct answer) or 0 (i.e., incorrect answer). |
| fit_predict_proba | Use the model to predict the responses in the testing data and returns the probabilities (that the correct answers will be provided). |
| eval | Predict learners' responses in the input val_data, and then return the AUC and Accuracy of the prediction. |
| save | Save the model to the given path. |
| load | Load the snapshot saved before from the given path. |
