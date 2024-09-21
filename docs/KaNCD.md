# KaNCD

The implementation of the KaNCD model in paper: [NeuralCD: A General Framework for Cognitive Diagnosis](https://ieeexplore.ieee.org/abstract/document/9865139)



If this code helps with your studies, please kindly cite the following publication:

```
@article{wang2022neuralcd,
  title={NeuralCD: A General Framework for Cognitive Diagnosis},
  author={Wang, Fei and Liu, Qi and Chen, Enhong and Huang, Zhenya and Yin, Yu and Wang, Shijin and Su, Yu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```



## Model description

KaNCD is an **K**nowledge-**a**ssociation based extension of the **N**eural**CD**M (alias NCDM in this package) model. In KaNCD, higher-order low dimensional latent traits of students, exercises and knowledge concepts are used respectively. 

The knowledge difficulty vector of an exercise is calculated from the latent trait of the exercise and the latent trait of each knowledge concept. 

![KDM_MF](_static\KDM_MF.png)

Similarly, the knowledge proficiency vector of a student is calculated from the latent trait of the student and the latent trait of each knowledge concept.

![KPM_MF](_static\KPM_MF.png)

Please refer to the paper for more details.

The model is implemented with Pytorch, and Adam optimizer is adopted for training.



## Parameters description

| Parameters | Type | Description                              |
| ---------- | ---- | ---------------------------------------- |
| meta_data  | dict | a dictionary containing all the userIds, itemIds, and skills. |
| dim        | int  | the dimension of the latent vectors of users, items and skills. default: 40 |
| mf_type    | str  | the type of layer(s) to be used for the interaction among latent vectors. default: "gmf".  choices: "mf", "gmf", "ncf1", "ncf2". |
| layer_dim1 | int  | the dimension of the first hidden layer. Default: 512 |
| layer_dim2 | int  | the dimension of the second hidden layer. Default: 256 |



### Examples

```python
import pandas as pd
from EduCDM import KaNCD
meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}
model = KaNCD(meta_data, 512, 256)

train_data = pd.DataFrame({'userId':[1,1,2,2,3,3], 'itemId': [1,2,1,3,2,3], 'skill': ["[1]", "[1,3]", "[1]", "[1,2,3]", "[1,3]", "[1,2,3]"], 'response': [1,1,0,1,1,0]})
test_data = pd.DataFrame({'userId':[1,2,3], 'itemId': [3,2,1], 'skill': ["[1,2,3]", "[1,3]", "[1]"], 'response': [1,1,0]})
model.fit(train_data, epoch=2)
predict = model.predict(test_data)
```



## Methods summary

| Methods           | Description                              |
| ----------------- | ---------------------------------------- |
| fit               | Fits the model to the training data.     |
| fit_predict       | Use the model to predict the responses in the testing data and returns the results. The responses are either 1 (i.e., correct answer) or 0 (i.e., incorrect answer). |
| fit_predict_proba | Use the model to predict the responses in the testing data and returns the probabilities (that the correct answers will be provided). |

