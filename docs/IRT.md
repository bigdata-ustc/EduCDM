# Item response theory

If the reader wants to know the details of EMIRT, please refer to the paper: *[Estimation for Item Response Models using the EM Algorithm for Finite Mixtures](https://files.eric.ed.gov/fulltext/ED405356.pdf)*.

```bibtex
@article{woodruff1996estimation,
  title={Estimation of Item Response Models Using the EM Algorithm for Finite Mixtures.},
  author={Woodruff, David J and Hanson, Bradley A},
  year={1996},
  publisher={ERIC}
}
```

## Introduction of model

![这是图片](_static\IRT\EMIRT\emirt1.png "Magic Gardens")

![这是图片](_static\IRT\EMIRT\emirt2.png "Magic Gardens")

![这是图片](_static\IRT\EMIRT\emirt3.png "Magic Gardens")

![这是图片](/assets/img/philly-magic-garden.jpg "Magic Gardens")

## Parameters description

| PARAMETERS | TYPE | DESCRIPTION                                                    |
| ---------- | ---- | -------------------------------------------------------------- |
| meta_data  | dict | a  dictionary containing all the userIds, itemIds, and skills. |
| dim        | int  | the  dimension of student's ability. Default: 1                |
| skip_value | int  | the skip_value of the item response matrix. Default: -1        |

##Examples

```python
import pandas as pd
from EduCDM import EMIRT
meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}
model = EMIRT(meta_data)

train_data = pd.DataFrame({'userId':[1,1,2,2,3,3], 'itemId': [1,2,1,3,2,3], 'skill': ["[1]", "[1,3]", "[1]", "[1,2,3]", "[1,3]", "[1,2,3]"], 'response': [1,1,0,1,1,0]})
test_data = pd.DataFrame({'userId':[1,2,3], 'itemId': [3,2,1], 'skill': ["[1,2,3]", "[1,3]", "[1]"], 'response': [1,1,0]})
model.fit(train_data, epoch=2)
predict = model.predict()
mrse, mse = model.eval(test_data)
```

## Methods summary

| METHODS           | DESCRIPTION                                                                                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| fit               | Fits  the model to the training data.                                                                                                                                   |
| fit_predict       | Use  the model to predict the responses in the testing data and returns the  results. The responses are either 1 (i.e., correct answer) or 0 (i.e.,  incorrect answer). |
| fit_predict_proba | Use  the model to predict the responses in the testing data and returns the  probabilities (that the correct answers will be provided).                                 |
