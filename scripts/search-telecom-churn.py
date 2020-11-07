#!/usr/bin/env python

from multiprocessing import Lock

from lightgbm_tuning import (
    parse_args,
    read_telecom_churn,
    wide_lightgbm_parameter_space,
    evaluate_lgb_experiment,
    run_pool,
    generate_random_experiments,
)

if __name__ == "__main__":
    args = parse_args()
    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn(
        args.n_folds, args.split_kind
    )

    # captures data and lock to use in forked processes
    def evaluator(experiment):
        evaluate_lgb_experiment(
            experiment,
            experiment_name=args.name,
            log_file=args.log,
            log_lock=log_lock,
            num_boost_round=args.num_boost_round,
            n_seeds=args.n_seeds,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        )

    run_pool(
        generator=generate_random_experiments(
            wide_lightgbm_parameter_space(), args.iterations
        ),
        evaluator=evaluator,
        processes=args.processes,
        chunksize=args.chunksize,
    )
