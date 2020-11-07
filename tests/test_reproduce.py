import multiprocessing
from functools import partial
import random
import pytest
from scipy.stats import randint
from scipy.stats import uniform
from lightgbm_tuning import (
    read_telecom_churn,
    evaluate_logreg_experiment,
    reproduce_logreg_experiment,
    generate_random_experiments,
    run_pool,
    loguniform,
    assert_logs_equal,
    wide_lightgbm_parameter_space,
    evaluate_lgb_experiment,
    log_generator,
    reproduce_lgb_experiment,
)


@pytest.mark.parametrize(
    "n_folds,split_kind", [(5, "k-folds"), (20, "shuffle-split")]
)
def test_logreg(n_folds, split_kind, tmp_path):
    manager = multiprocessing.Manager()
    log_lock = manager.Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn(
        n_folds, split_kind
    )

    iterations = 50
    experiment_log = tmp_path / "experiment.log"
    parameter_space = {
        "clf__C": loguniform(-5, 3, 10),
        "clf__class_weight": loguniform(-2, 2, 10),
        "clf__dual": [False],
        "clf__fit_intercept": [True, False],
        "clf__intercept_scaling": [1],
        "clf__max_iter": [500],
        "clf__multi_class": ["ovr"],
        "clf__n_jobs": [1],
        "clf__penalty": ["l2"],
        "clf__random_state": randint(1, 100000),
        "clf__solver": ["lbfgs"],
        "clf__tol": loguniform(-10, -2, 10),
        "clf__verbose": [0],
        "clf__warm_start": [False],
    }

    run_pool(
        generator=generate_random_experiments(parameter_space, iterations),
        evaluator=partial(
            evaluate_logreg_experiment,
            experiment_name="reproduce_logreg",
            n_seeds=3,
            log_file=experiment_log,
            log_lock=log_lock,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        ),
        processes=4,
        chunksize=5,
    )

    random.seed(42)
    shuffled_log = list(log_generator(experiment_log))
    random.shuffle(shuffled_log)

    reproduce_log = tmp_path / "reproduce.log"

    run_pool(
        generator=shuffled_log,
        evaluator=partial(
            reproduce_logreg_experiment,
            log_file=reproduce_log,
            log_lock=log_lock,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        ),
        processes=4,
        chunksize=5,
    )

    assert_logs_equal(experiment_log, reproduce_log, n_folds)


@pytest.mark.parametrize(
    "n_folds,split_kind", [(5, "k-folds"), (20, "shuffle-split")]
)
def test_lightgbm(n_folds, split_kind, tmp_path):
    manager = multiprocessing.Manager()
    log_lock = manager.Lock()
    X_train, X_val, y_train, y_val, folds = read_telecom_churn(
        n_folds, split_kind
    )

    iterations = 100
    num_boost_round = 50
    experiment_log = tmp_path / "experiment.log"
    parameter_space = wide_lightgbm_parameter_space()

    run_pool(
        generator=generate_random_experiments(parameter_space, iterations),
        evaluator=partial(
            evaluate_lgb_experiment,
            experiment_name="reproduce_lightgbm",
            n_seeds=3,
            num_boost_round=num_boost_round,
            log_file=experiment_log,
            log_lock=log_lock,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        ),
        processes=4,
        chunksize=5,
    )

    random.seed(42)
    shuffled_log = list(log_generator(experiment_log))
    random.shuffle(shuffled_log)

    reproduce_log = tmp_path / "reproduce.log"

    run_pool(
        generator=shuffled_log,
        evaluator=partial(
            reproduce_lgb_experiment,
            log_file=reproduce_log,
            log_lock=log_lock,
            num_boost_round=num_boost_round,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            folds=folds,
        ),
        processes=4,
        chunksize=5,
    )

    assert_logs_equal(experiment_log, reproduce_log, n_folds)
