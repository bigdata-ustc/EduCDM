# Item response theory

EMIRT in this package denotes the IRT model trained with EM algorithm. If the reader wants to know the details of EMIRT, please refer to the paper: *[Estimation for Item Response Models using the EM Algorithm for Finite Mixtures](https://files.eric.ed.gov/fulltext/ED405356.pdf)*.

If this code helps you, please cite our work

```bibtex
@misc{bigdata2024educdm,
  title={EduCDM},
  author={bigdata-ustc},
  publisher = {GitHub},
  journal = {GitHub repository},
  year = {2024},
  howpublished = {\url{https://github.com/bigdata-ustc/EduCDM}},
}
```

## Brief Introduction to IRT

Item response theory (IRT) is one of the most representative model for cognitive diagnosis. IRT uses parameters to represent students' abilities and the traits of exercises (e.g., difficulty, discrimination, guess). In this EMIRT, we implement the three-parameter logistic model whose item response function is as follows:

![model](_static\IRT\EMIRT\emirt4.png "Magic Gardens")

In EMIRT, EM algorithm is adopted to estimate the parameters.



## Parameters description

| PARAMETERS | TYPE | DESCRIPTION                              |
| ---------- | ---- | ---------------------------------------- |
| meta_data  | dict | a  dictionary containing all the userIds, itemIds, and skills. |
| dim        | int  | the  dimension of student's ability. Default: 1 |
| skip_value | int  | the skip_value of the item response matrix. Default: -1 |

## Examples

```python
import pandas as pd
from EduCDM import EMIRT
meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}
model = EMIRT(meta_data)

train_data = pd.DataFrame({'userId':[1,1,2,2,3,3], 'itemId': [1,2,1,3,2,3], 'skill': ["[1]", "[1,3]", "[1]", "[1,2,3]", "[1,3]", "[1,2,3]"], 'response': [1,1,0,1,1,0]})
test_data = pd.DataFrame({'userId':[1,2,3], 'itemId': [3,2,1], 'skill': ["[1,2,3]", "[1,3]", "[1]"], 'response': [1,1,0]})
model.fit(train_data, epoch=2)
predict = model.predict()
auc, acc = model.eval(test_data)
```

## Methods summary

| METHODS           | DESCRIPTION                              |
| ----------------- | ---------------------------------------- |
| fit               | Fits  the model to the training data.    |
| fit_predict       | Use  the model to predict the responses in the testing data and returns the  results. The responses are either 1 (i.e., correct answer) or 0 (i.e.,  incorrect answer). |
| fit_predict_proba | Use  the model to predict the responses in the testing data and returns the  probabilities (that the correct answers will be provided). |
| eval | Predict learners' responses in the input val_data, and then return the AUC and Accuracy of the prediction. |
| save | Save the model to the given path. |
| load | Load the snapshot saved before from the given path. |
