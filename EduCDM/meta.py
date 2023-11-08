# coding: utf-8
# 2023/11/4 @ Fei Wang

from typing import List


def etl(*args, **kwargs) -> ...:  # pragma: no cover
    """
    extract - transform - load
    """
    pass


def train(*args, **kwargs) -> ...:  # pragma: no cover
    pass


def evaluate(*args, **kwargs) -> ...:  # pragma: no cover
    pass


class CDM(object):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> List[float]:
        raise NotImplementedError

    def predict_proba(self, *args, **kwargs) -> List[float]:
        raise NotImplementedError

    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError
