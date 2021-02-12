# -*- coding: utf-8 -*-
# third party
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

# ite relative
from . import utils  # noqa: F401

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "ite"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
