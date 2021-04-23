# coding: utf-8
# 2021/4/23 @ tongshiwei

import numpy as np

__all__ = ["irf", "irt3pl"]


def irf(theta, a, b, c, D=1.702, *, F=np):
    return c + (1 - c) / (1 + F.exp(-D * a * (theta - b)))


irt3pl = irf
