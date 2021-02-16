# third party
import pytest

# ite absolute
import ite.algs.hyperparam_tuning as tuning


@pytest.mark.parametrize("ganite_ver", ["GANITE", "GANITE_TORCH"])
@pytest.mark.slow
def test_hyperparam_tuning(ganite_ver: str) -> None:
    tuning.search(ganite_ver, 1)
