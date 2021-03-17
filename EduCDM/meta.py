# coding: utf-8
# 2021/3/17 @ tongshiwei


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
    def __init__(self, *args, **kwargs) -> ...:
        pass

    def train(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def save(self, *args, **kwargs) -> ...:
        raise NotImplementedError

    def load(self, *args, **kwargs) -> ...:
        raise NotImplementedError
