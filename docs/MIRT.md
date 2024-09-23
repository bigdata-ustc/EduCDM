# Multidimensional Item Response Theory

If the reader wants to know the details of MIRT, please refer to the paper: *[Multidimensional item response theory models](http://ndl.ethernet.edu.et/bitstream/123456789/60415/1/116.pdf)*

```
@incollection{reckase2009multidimensional,
  title={Multidimensional item response theory models},
  author={Reckase, Mark D},
  booktitle={Multidimensional item response theory},
  pages={79--112},
  year={2009},
  publisher={Springer}
}
```

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

Multidimensional Item Response Theory (MIRT) is the multidimensional extension of IRT. The extensions varies in different specific models. In this package, we implement a wide-accepted version, where the item response function is as follows:
$$
Pr(U_{ij}=1) = \frac{1}{1 + \exp^{-\bm a_j \cdot \bm \theta_i + \bm b_j}},
$$
where $\bm \theta_i$ indicates the multidimensional latent ability of learner $i$, $\bm a_j$ and $\bm b_j$ are related to the difficulty and discrimination of item $j$.

## Parameters description

| PARAMETERS | TYPE | DESCRIPTION                              |
| ---------- | ---- | ---------------------------------------- |
| meta_data  | dict | a  dictionary containing all the userIds, itemIds, and skills. |
| latent_dim | int  | the  dimension of ability parameter. Default: 20. |

## Examples

```python
import pandas as pd
from EduCDM import MIRT
meta_data = {'userId': ['001', '002', '003'], 'itemId': ['adf', 'w5'], 'skill': ['skill1', 'skill2', 'skill3', 'skill4']}
model = MIRT(meta_data, 20)

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

