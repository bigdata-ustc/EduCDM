# Deterministic Inputs, Noisy “And” gate model
The implementation of the classical cognitive diagnosis model, i.e., DINA (Deterministic Inputs, Noisy “And” gate) model. The training process is adapted to using gradient descending methods. If the reader wants to know the details of DINA, please refer to the Appendix of the paper: *[DINA model and parameter estimation: A didactic](https://journals.sagepub.com/doi/10.3102/1076998607309474)*.

If this repository is helpful for you, please cite our work

```
@misc{bigdata2021educdm,
  title={EduCDM},
  author={bigdata-ustc},
  publisher = {GitHub},
  journal = {GitHub repository},
  year = {2021},
  howpublished = {\url{https://github.com/bigdata-ustc/EduCDM}},
}
```

## Model description
DINA model is a classical cognitive diagnosis model, where each learner $i$ is represented with a binary vector ($[\alpha_{ik}, k=1,2,...K]$ in the following figure) indicating the learner's knowledge mastery pattern. A Q-matrix $Q=\{0, 1\}^{J\times K}$ indicates relevant skills (or knowledge components) of the test items. For each test item $j$, there are possibilities to slip on it and guess the correct answer, which are characterized by the parameters $s_j$ and $g_j$ respectively. Overall, the probability that learner $i$ would provide a correct response to item $j$ is calculated as follows:
$$Pr(X_{ij}=1|\alpha_i,q_j, s_j, g_j) = (1-s_j)^{\eta_{ij}}g_j^{1-\eta_{ij}},$$

$$
\eta_{ij} = \prod_{k=1}^{K}\alpha_{ik}^{q_{jk}}.
$$

<img src=_static/DINA.png width=90%>

## Parameters description

| Parameters | Type | Description                              |
| ---------- | ---- | ---------------------------------------- |
| meta_data  | dict | a dictionary containing all the userIds, itemIds, and skills. |
| max_slip        | float  | the maximum value of possible slipping. default: 0.4 |
| max_guess    | float  | the maximum value of possible slipping. default: 0.4 |


## Methods summary

| Methods           | Description                              |
| ----------------- | ---------------------------------------- |
| fit               | Fits the model to the training data.     |
| fit_predict       | Use the model to predict the responses in the testing data and returns the results. The responses are either 1 (i.e., correct answer) or 0 (i.e., incorrect answer). |
| fit_predict_proba | Use the model to predict the responses in the testing data and returns the probabilities (that the correct answers will be provided). |

