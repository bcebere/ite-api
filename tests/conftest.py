# stdlib
import os
from typing import Any
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def warn(*args: Any, **kwargs: Any) -> None:
    pass


warnings.warn = warn
