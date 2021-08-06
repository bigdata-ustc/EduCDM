<p align="center">
  <img width="300" src="docs/_static/EduCDM.png">
</p>

# EduCDM


[![PyPI](https://img.shields.io/pypi/v/EduCDM.svg)](https://pypi.python.org/pypi/EduCDM)
[![test](https://github.com/bigdata-ustc/EduCDM/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/bigdata-ustc/EduCDM/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/bigdata-ustc/EduCDM/branch/main/graph/badge.svg?token=B7gscOGQLD)](https://codecov.io/gh/bigdata-ustc/EduCDM)
[![Download](https://img.shields.io/pypi/dm/EduCDM.svg?style=flat)](https://pypi.python.org/pypi/EduCDM)
[![License](https://img.shields.io/github/license/bigdata-ustc/EduCDM)](LICENSE)
[![DOI](https://zenodo.org/badge/348569904.svg)](https://zenodo.org/badge/latestdoi/348569904)

The Model Zoo of Cognitive Diagnosis Models, including classic Item Response Ranking (**IRT**), Multidimensional Item Response Ranking (**MIRT**), Deterministic Input, Noisy "And" model(**DINA**), and advanced Fuzzy Cognitive Diagnosis Framework (**FuzzyCDF**), Neural Cognitive Diagnosis Model (**NCDM**) and Item Response Ranking framework (**IRR**).

## Brief introduction to CDM

Cognitive diagnosis model (CDM) for intelligent educational systems is a type of  model that infers students' knowledge states from their learning behaviors (especially exercise response logs). 



Typically, the input of a CDM could be the students' response logs of items (i.e., exercises/questions), the Q-matrix that denotes the correlation between items and knowledge concepts (skills). The output is the diagnosed student knowledge states, such as students' abilities and students' proficiencies on each knowledge concepts.



Traditional CDMs include:

- [IRT](https://link.springer.com/book/10.1007/978-0-387-89976-3): item response theory, a continuous unidimensional CDM with logistic-like item response function.
- [MIRT](https://link.springer.com/book/10.1007/978-0-387-89976-3): Multidimensional item response theory, a continuous multidimensional CDM with logistic-like item response function. Mostly extended from unidimensional IRT.
- [DINA](https://journals.sagepub.com/doi/10.3102/1076998607309474): deterministic input, noisy "and" model, a discrete multidimensional CDM. Q-matrix is used to model the effect of knowledge concepts in the cognitive process, as well as guessing and slipping factors.

etc.

More recent researches about CDMs:

- [FuzzyCDF](http://staff.ustc.edu.cn/~qiliuql/files/Publications/Qi-Liu-TIST2018.pdf): fuzzy cognitive diagnosis framework, a continuous multidimensional CDM for students' cognitive modeling with both objective and subjective items.
- [NeuralCD](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2020/Fei-Wang-AAAI.pdf): neural cognitive diagnosis framework, a neural-network-based general cognitive diagnosis framework. In this repository we provide the basic implementation NCDM.
- [IRR](http://home.ustc.edu.cn/~tongsw/files/IRR.pdf): item response ranking framework, a pairwise cognitive diagnosis framework. In this repository we provide the several implementations for most of CDMs.

## List of models

* [NCDM](EduCDM/NCDM) [[doc]](docs/NCDM.md) [[example]](examples/NCDM)
* [FuzzyCDF](EduCDM/FuzzyCDF) [[doc]](docs/FuzzyCDF.md) [[example]](examples/FuzzyCDF)
* [DINA](EduCDM/DINA) [[doc]](docs/DINA.md) [[example]](examples/DINA)
  * Eexpectation Maximization ([EMDINA](EduCDM/DINA/EM)) [[example]](examples/DINA/EM)
  * Gradient Descent ([GDDINA](EduCDM/DINA/GD)) [[example]](examples/DINA/GD)
* [MIRT](EduCDM/MIRT) [[doc]](docs/MIRT.md) [[example]](examples/MIRT)
* [IRT](EduCDM/IRT) [[doc]](docs/IRT.md) [[example]](examples/IRT)
  * Eexpectation Maximization ([EMIRT](EduCDM/IRT/EM)) [[example]](examples/IRT/EM)
  * Gradient Descent ([GDIRT](EduCDM/IRT/GD)) [[example]](examples/IRT/GD)
* [MCD](EduCDM/MCD) [[doc]](docs/MCD.md) [[example]](examples/MCD)
* [IRR](EduCDM/IRR) [[doc]](docs/IRR.md)[[example]](examples/IRR)
  * [IRR-NCDM](examples/IRR/NCDM.ipynb)
  * [IRR-MIRT](examples/IRR/MIRT.ipynb)
  * [IRR-DINA](examples/IRR/DINA.ipynb)
  * [IRR-IRT](examples/IRR/IRT.ipynb)

## Installation

Git and install with `pip`:

```
git clone https://github.com/bigdata-ustc/EduCDM.git
cd path/to/code
pip install .
```

Or directly install from pypi:

```
pip install EduCDM
```


## Contribute

EduCDM is still under development. More algorithms and features are going to be added and we always welcome contributions to help make EduCDM better. If you would like to contribute, please follow this [guideline](CONTRIBUTE.md).

## Citation

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
