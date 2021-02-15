# -*- coding: utf-8 -*-
# stdlib
import os
from typing import Any
import warnings

# third party
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution


def warn(*args: Any, **kwargs: Any) -> None:
    pass


warnings.warn = warn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ite absolute
import ite.algs  # noqa: F401,E402
import ite.datasets  # noqa: F401,E402
import ite.utils  # noqa: F401,E402

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "ite"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
