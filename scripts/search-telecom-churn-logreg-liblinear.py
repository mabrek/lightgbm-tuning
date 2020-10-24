#!/usr/bin/env python

from multiprocessing import Lock

from scipy.stats import randint as randint
from scipy.stats import uniform as uniform

from lightgbm_tuning import (
    loguniform,
    parse_args,
    read_telecom_churn,
    evaluate_logreg_experiment,
    run_pool,
    generate_random_experiments,
)

if __name__ == "__main__":
    parameter_space = {
        "clf__C": loguniform(-5, 3, 10),
        "clf__class_weight": loguniform(-2, 2, 10),
        "clf__dual": [True, False],
        "clf__fit_intercept": [True, False],
        "clf__intercept_scaling": uniform(loc=0.01, scale=99.99),
        "clf__max_iter": [10000],
        "clf__multi_class": ["warn"],
        "clf__n_jobs": [1],
        "clf__penalty": ["l1", "l2"],
        "clf__random_state": randint(1, 30000),
        "clf__solver": ["liblinear"],
        "clf__tol": loguniform(-10, -2, 10),
        "clf__verbose": [0],
        "clf__warm_start": [False],
    }

    args = parse_args()
    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn(
        args.n_folds, args.split_kind
    )

    # captures data and lock to use in forked processes
    def evaluator(experiment):
        evaluate_logreg_experiment(
            experiment,
            experiment_name=args.name,
            n_seeds=args.n_seeds,
            log_file=args.log,
            log_lock=log_lock,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        )

    run_pool(
        generate_random_experiments(parameter_space, args.iterations),
        args,
        evaluator,
    )
