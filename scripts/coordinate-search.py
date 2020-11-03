#!/usr/bin/env python

from multiprocessing import Lock
import argparse

import numpy as np
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler

from lightgbm_tuning import (
    read_json_log,
    read_telecom_churn,
    run_pool,
    evaluate_lgb_parameters,
    log_json,
    loguniform,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Explore all coordinates of parameter set"
    )
    parser.add_argument("--input-log", required=True)
    parser.add_argument("--output-log", required=True)
    parser.add_argument("--coordinate-iterations", type=int, default=100)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--split-kind",
        type=str,
        default="k-folds",
        choices=["k-folds", "shuffle-split"],
    )
    parser.add_argument("--num-boost-round", type=int, default=500)
    args = parser.parse_args()

    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn(
        args.n_folds, args.split_kind
    )

    input_log = read_json_log(args.input_log)

    coordinates = {
        "learning_rate": loguniform(low=-8, high=6, base=10),
        "num_leaves": randint(2, 4000),
        "max_depth": randint(1, 400),
        "min_data_in_leaf": randint(1, 2000),
        "min_sum_hessian_in_leaf": loguniform(low=-10, high=6, base=10),
        "bagging_enable": [False, True],
        "bagging_fraction": uniform(loc=0.1, scale=0.9),
        "bagging_freq": randint(1, 50),
        "feature_fraction": uniform(loc=0.2, scale=0.8),
        "max_delta_step": loguniform(low=-8, high=6, base=10),
        "lambda_l1": loguniform(low=-10, high=6, base=10),
        "lambda_l2": loguniform(low=-10, high=10, base=10),
        "min_gain_to_split": loguniform(low=-10, high=6, base=10),
        "min_data_per_group": randint(1, 4000),
        "max_cat_threshold": randint(1, 2000),
        "cat_l2": loguniform(low=-10, high=10, base=10),
        "cat_smooth": loguniform(low=-10, high=10, base=10),
        "max_cat_to_onehot": randint(1, 100),
        "scale_pos_weight": uniform(loc=0.1, scale=99.9),
        "boost_from_average": [False, True],
        "max_bin": randint(4, 2048),
        "min_data_in_bin": randint(1, 5000),
        "bin_construct_sample_cnt": randint(5, 10000),
    }

    def generator():
        experiment_id = 0
        for _, row in input_log.iterrows():
            base = (
                row.filter(regex="^param_")
                .rename(lambda x: x.replace("param_", ""))
                .to_dict()
            )
            if np.isnan(base["scale_pos_weight"]):
                base["scale_pos_weight"] = None
            base = {k: [v] for k, v in base.items()}

            for c, c_range in coordinates.items():
                subspace = base.copy()
                subspace["seed"] = randint(1, 100000)
                subspace[c] = c_range
                if c == "scale_pos_weight":
                    subspace["is_unbalance"] = [False]
                for parameters in ParameterSampler(
                    subspace, args.coordinate_iterations
                ):
                    experiment_id += 1
                    yield (row["name"], experiment_id, parameters)

    def evaluator(experiment):
        name, experiment_id, parameters = experiment
        log_data = {}
        log_data["name"] = name
        log_data["experiment_id"] = experiment_id
        log_data.update({"param_" + k: v for k, v in parameters.items()})
        metrics = evaluate_lgb_parameters(
            parameters,
            num_boost_round=args.num_boost_round,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        )
        metrics["success"] = True
        log_data.update(metrics)
        with log_lock:
            log_json(args.output_log, log_data)

    run_pool(generator(), args, evaluator)
