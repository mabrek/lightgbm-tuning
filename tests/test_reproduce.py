import multiprocessing
from functools import partial
import pytest
from scipy.stats import randint as randint
from lightgbm_tuning import (
    read_telecom_churn,
    evaluate_logreg_experiment,
    reproduce_logreg_experiment,
    evaluate_logreg_parameters,
    generate_random_experiments,
    run_pool,
    read_json_log,
    log_json,
    loguniform,
    assert_logs_equal,
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

    iterations = 100
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
        processes=3,
        chunksize=1,
    )

    reproduce_log = tmp_path / "reproduce.log"

    def log_generator():
        for _, row in read_json_log(experiment_log).iterrows():
            parameters = (
                row.filter(regex="^param_")
                .rename(lambda x: x.replace("param_", ""))
                .to_dict()
            )
            yield (row["name"], row["experiment_id"], parameters)

    run_pool(
        generator=log_generator(),
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
        processes=2,
        chunksize=1,
    )

    assert_logs_equal(experiment_log, reproduce_log, n_folds)
