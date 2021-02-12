"""
Example/Test script for ganite.
"""
# stdlib
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# third party
import initpath_alg
import utilmlab

initpath_alg.init_sys_path()


def init_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", help="python interpreter to use")
    parser.add_argument("--it", default=10000, type=int)
    parser.add_argument("--kk", default=1)
    parser.add_argument("--alpha")
    parser.add_argument("-o")
    parser.add_argument("--projdir")
    parser.add_argument("--verify", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":

    args = init_arg()

    verify = args.verify

    proj_dir = utilmlab.get_proj_dir() if args.projdir is None else args.projdir

    fn_json_result_test = args.o

    alg = "ganite"

    if args.exe is not None:
        python_exe = args.exe
    else:
        python_exe = "python" if sys.version_info[0] < 3 else "python3"

    niter = args.it
    version = 3
    arg_alpha = args.alpha
    kk = args.kk

    resdir = "{}/result/{}/v_{}/h_{}".format(
        proj_dir,
        alg,
        version,
        os.environ["HOSTNAME"] if "HOSTNAME" in os.environ else "unknown",
    )

    utilmlab.ensure_dir(resdir)

    logger = utilmlab.init_logger(resdir, f"log_mtest_{alg}_ana.txt")

    result_lst = []

    dataset_lst = [("twins", 0.1), ("jobs", 0.5)]

    script_data = Path(f"{proj_dir}/alg/ganite/data_preprocessing_ganite.py")
    script = Path(f"{proj_dir}/alg/ganite/ganite.py")
    script_ana = Path(f"{proj_dir}/alg/ganite/ganite_ana.py")

    for dataset_tpl in dataset_lst:

        dataset, alpha = dataset_tpl

        alpha = alpha if arg_alpha is None else arg_alpha

        odir = f"{resdir}/dataset_{dataset}"

        fn_trainx = f"{odir}/trainx.csv"
        fn_trainy = f"{odir}/trainy.csv"
        fn_traint = f"{odir}/traint.csv"
        fn_testx = f"{odir}/testx.csv"
        fn_testy = f"{odir}/testy.csv"
        fn_testt = f"{odir}/testt.csv"
        fn_yhat = f"{odir}/yhat.csv"

        if os.path.isfile(fn_testt):
            os.remove(fn_testt)

        utilmlab.ensure_dir(odir)

        fn_json_result_dataset = f"{odir}/result.json"
        fn_json_result_dataset_ana = f"{odir}/result_ana.json"

        utilmlab.exe_cmd(
            logger,
            "{} {} --dataset {} "
            "--trainx {} --trainy {} --traint {} "
            "--testx {} --testy {} --testt {}".format(
                python_exe,
                script_data,
                dataset,
                fn_trainx,
                fn_trainy,
                fn_traint,
                fn_testx,
                fn_testy,
                fn_testt,
            ),
        )

        arg_testt = f"--testt {fn_testt}" if os.path.isfile(fn_testt) else ""
        arg_reftreatment = (
            f"--ref_treatment {fn_testt}" if os.path.isfile(fn_testt) else ""
        )

        utilmlab.exe_cmd(
            logger,
            "{} {} -o {} --alpha {} "
            " --it {} --kk {} -ocsv {} "
            "--trainx {} --trainy {} --traint {} "
            "--testx {} --testy {} {}".format(
                python_exe,
                script,
                fn_json_result_dataset,
                alpha,
                niter,
                kk,
                fn_yhat,
                fn_trainx,
                fn_trainy,
                fn_traint,
                fn_testx,
                fn_testy,
                arg_testt,
            ),
        )

        with open(fn_json_result_dataset) as f:
            result_d = json.load(f)

        result_d["dataset"] = dataset

        utilmlab.exe_cmd(
            logger,
            "{} {} -o {} -i {} --ref {} {}".format(
                python_exe,
                script_ana,
                fn_json_result_dataset_ana,
                fn_yhat,
                fn_testy,
                arg_reftreatment,
            ),
        )

        with open(fn_json_result_dataset_ana) as f:
            result_d_ana = json.load(f)

        result_d["ana"] = result_d_ana
        result_lst.append(result_d)

    logger.info(f"\n\nOverview result {alg}:\n")

    for el in result_lst:
        logger.info(f"{el}")

    if fn_json_result_test is not None:
        d = {"result": result_lst}
        with open(fn_json_result_test, "w") as fp:
            json.dump(d, fp)

    if verify:
        for el in result_lst:
            if el["dataset"] == "twins":
                res = el["ana"]
                assert 0.28 < res["sqrt_PEHE"] and res["sqrt_PEHE"] < 0.31
