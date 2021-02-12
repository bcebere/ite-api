# stdlib
import argparse
import json
from typing import Any

# third party
import numpy as np
import pandas as pd

# ite relative
from ..utils.tensorflow import ATE
from ..utils.tensorflow import PEHE
from ..utils.tensorflow import Perf_RPol_ATT


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-o")
    parser.add_argument("--ref")
    parser.add_argument("--ref_treatment")
    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    fn_json = args.o

    df_hat = pd.read_csv(args.i)
    df_ref = pd.read_csv(args.ref)
    df_ref_treatment = (
        None if args.ref_treatment is None else pd.read_csv(args.ref_treatment)
    )
    y_hat = np.argmax(df_hat.values, axis=1)
    y_ref = np.argmax(df_ref.values, axis=1)

    result = dict()
    if df_ref_treatment is None:
        result["sqrt_PEHE"] = float(np.sqrt(PEHE(df_ref.values, df_hat.values)))
        result["ATE"] = float(ATE(df_ref.values, df_hat.values))
    else:
        result["Perf_RPol_ATT"] = float(
            Perf_RPol_ATT(df_ref_treatment.values, df_ref.values, df_hat.values)[0]
        )
    print(result)
    with open(fn_json, "w") as fp:
        json.dump(result, fp)
