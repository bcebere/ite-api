# -*- coding: utf-8 -*-

import pytest

from ite.skeleton import fib

__author__ = "Cebere Bogdan"
__copyright__ = "Cebere Bogdan"
__license__ = "mit"


def test_fib() -> None:
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
