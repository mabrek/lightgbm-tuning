__all__ = [
    "loguniform",
    "read_json_log",
    "log_json",
    "summarize_logs",
    "check_omitted_parameters",
    "EVAL_AT",
    "METRICS",
    "SUBSET_METRICS",
    "WHOLE_METRICS",
    "CONT_PARAMETERS",
    "LOG_PARAMETERS",
    "SET_PARAMETERS",
    "INT_PARAMETERS",
    "drop_boring_columns",
    "read_full_logs",
    "unfold_iterations",
    "shaderdots",
    "top_mean_dev_auc",
    "top_min_dev_auc",
    "top_min_whole_validation_auc",
    "parse_args",
    "read_telecom_churn",
    "exclude_columns",
    "read_files",
    "summarize_to_chunks",
    "aggregate_chunks",
]

import warnings
import traceback
import sys
from datetime import datetime
import json
from itertools import product, chain, islice
from multiprocessing import Pool, get_context
import logging
import argparse
from functools import partial
import gc
from glob import glob
import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from sklearn.utils import check_random_state
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
import lightgbm as lgb

try:
    from bokeh.plotting import figure
except ImportError as e:
    warnings.warn(
        f'exception "{e}" while importing bokeh, plotting will be unavailable'
    )

try:
    import datashader as ds
    import datashader.transfer_functions as tf
    from datashader.bokeh_ext import InteractiveImage
except ImportError as e:
    warnings.warn(
        f'exception "{e}" while importing datashader, plotting will be unavailable'
    )


logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm.basic")

EVAL_AT = [10, 100, 1000]

METRICS = ["binary_logloss", "auc", "binary_error", "kldiv"] + [
    f"map_{i}" for i in EVAL_AT
]

SUBSET_METRICS = [
    "_".join([d, m])
    for d, m in product(["train", "dev", "validation"], METRICS)
]

WHOLE_METRICS = [
    "_".join([d, m])
    for d, m in product(["whole_train", "whole_validation"], METRICS)
]

CONT_PARAMETERS = [
    "param_bagging_fraction",
    "param_feature_fraction",
    "param_scale_pos_weight",
    "iteration",
]

LOG_PARAMETERS = [
    "param_learning_rate",
    "param_min_sum_hessian_in_leaf",
    "param_max_delta_step",
    "param_lambda_l1",
    "param_lambda_l2",
    "param_min_gain_to_split",
    "param_cat_l2",
    "param_cat_smooth",
]

SET_PARAMETERS = ["param_is_unbalance", "param_boost_from_average"]

INT_PARAMETERS = [
    "param_num_leaves",
    "param_max_depth",
    "param_min_data_in_leaf",
    "param_bagging_freq",
    "param_min_data_per_group",
    "param_max_cat_threshold",
    "param_max_cat_to_onehot",
    "param_max_bin",
    "param_min_data_in_bin",
    "param_bin_construct_sample_cnt",
]

LOGREG_LOG_PARAMETERS = [
    "param_clf__C",
    "param_clf__class_weight",
    "param_clf__tol",
]

LOGREG_CONT_PARAMETERS = ["param_clf__intercept_scaling"]


def read_telecom_churn(n_folds, split_kind, random_state=67345):
    # TODO move data path to config
    df = pd.read_csv(
        "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        index_col="customerID",
        dtype={
            "gender": "category",
            "SeniorCitizen": "category",
            "Partner": "category",
            "Dependents": "category",
            "PhoneService": "category",
            "MultipleLines": "category",
            "InternetService": "category",
            "OnlineSecurity": "category",
            "OnlineBackup": "category",
            "DeviceProtection": "category",
            "TechSupport": "category",
            "StreamingTV": "category",
            "StreamingMovies": "category",
            "Contract": "category",
            "PaperlessBilling": "category",
            "PaymentMethod": "category",
        },
    ).sample(frac=1, random_state=102984)
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")

    y = (df.Churn == "Yes").astype(np.int8)
    X = df.drop(["Churn", "tenure", "TotalCharges"], axis="columns")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=834936
    )

    if split_kind == "k-folds":
        folds = list(
            StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=random_state
            ).split(X_train, y_train)
        )
    elif split_kind == "shuffle-split":
        folds = list(
            StratifiedShuffleSplit(
                n_splits=n_folds, test_size=0.2, random_state=random_state
            ).split(X_train, y_train)
        )

    return X_train, X_val, y_train, y_val, folds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random search LightGBM parameters"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument(
        "--split-kind",
        type=str,
        default="k-folds",
        choices=["k-folds", "shuffle-split"],
    )
    parser.add_argument("--num-boost-round", type=int, default=500)
    return parser.parse_args()


def generate_random_experiments(parameter_space, iterations):
    return enumerate(ParameterSampler(parameter_space, iterations))


def run_pool(generator, evaluator, processes, chunksize=10, verbose=False):
    with get_context("fork").Pool(processes=processes) as pool:
        results = pool.imap_unordered(evaluator, generator, chunksize=chunksize)
        for _ in results:
            if verbose:
                print(".", end="", flush=True)


# from https://github.com/scikit-learn/scikit-learn/blob/19bffee9b172cf169fded295e3474d1de96cdc57/sklearn/utils/random.py
class loguniform:
    """A class supporting log-uniform random variables.
    Parameters
    ----------
    low : float
        The log minimum value
    high : float
        The log maximum value
    base : float
        The base for the exponent.
    Methods
    -------
    rvs(self, size=None, random_state=None)
        Generate log-uniform random variables
    Notes
    -----
    This class generates values between ``base**low`` and ``base**high`` or
        base**low <= loguniform(low, high, base=base).rvs() <= base**high
    The logarithmic probability density function (PDF) is uniform. When
    ``x`` is a uniformly distributed random variable between 0 and 1, ``10**x``
    are random variales that are equally likely to be returned.
    """

    def __init__(self, low, high, base=10):
        """
        Create a log-uniform random variable.
        """
        self._low = low
        self._high = high
        self._base = base

    def rvs(self, size=None, random_state=None):
        """
        Generates random variables with ``base**low <= rv <= base**high``
        where ``rv`` is the return value of this function.
        Parameters
        ----------
        size : int or tuple, optional
            The size of the random variable.
        random_state : int, RandomState, optional
            A seed (int) or random number generator (RandomState).
        Returns
        -------
        rv : float or np.ndarray
            Either a single log-uniform random variable or an array of them
        """
        _rng = check_random_state(random_state)
        unif = _rng.uniform(self._low, self._high, size=size)
        rv = np.power(self._base, unif)
        return rv


def read_json_log(f, chunksize=None):
    return pd.read_json(
        f,
        typ="frame",
        orient="records",
        lines=True,
        precise_float=True,
        chunksize=chunksize,
        dtype={"success": "object"},
    )


def log_json(file, data):
    with open(file, "at") as output:
        data["timestamp"] = datetime.now().isoformat()
        print(json.dumps(data), file=output, flush=True)
        os.fsync(output.fileno())


def evaluate_lgb_experiment(
    experiment,
    experiment_name,
    log_file,
    log_lock,
    num_boost_round,
    n_seeds,
    X_train,
    X_val,
    y_train,
    y_val,
    folds,
):

    experiment_id, parameters = experiment

    if parameters["is_unbalance"]:
        parameters["scale_pos_weight"] = None

    if "bagging_enable" in parameters:
        if not parameters["bagging_enable"]:
            parameters["bagging_fraction"] = 1
            parameters["bagging_freq"] = 0
        del parameters["bagging_enable"]

    if "feature_fraction_enable" in parameters:
        if not parameters["feature_fraction_enable"]:
            parameters["feature_fraction"] = 1
        del parameters["feature_fraction_enable"]

    log_data = {}
    log_data["name"] = experiment_name
    log_data["experiment_id"] = experiment_id

    root_seed = parameters["seed"]
    for sub_seed in range(n_seeds):
        parameters["seed"] = root_seed + sub_seed
        try:
            log_data.update({"param_" + k: v for k, v in parameters.items()})
            metrics = evaluate_lgb_parameters(
                parameters,
                num_boost_round,
                X_train,
                X_val,
                y_train,
                y_val,
                folds,
            )
            metrics["success"] = True
            log_data.update(metrics)

        except Exception as e:
            warnings.warn(f'got Exception "{e}" for parameters {parameters}')
            traceback.print_exc(file=sys.stderr)  # TODO use logger instead
        finally:
            with log_lock:
                log_json(log_file, log_data)


def evaluate_lgb_parameters(
    parameters, num_boost_round, X_train, X_val, y_train, y_val, folds
):

    whole_train = lgb.Dataset(
        X_train, label=y_train, group=[X_train.shape[0]], free_raw_data=False
    )
    validation = lgb.Dataset(
        X_val, label=y_val, group=[X_val.shape[0]], free_raw_data=False
    )

    metrics = {}
    for f in range(len(folds)):
        train_idx, dev_idx = folds[f]
        train = lgb.Dataset(
            X_train.iloc[train_idx],
            label=y_train.iloc[train_idx],
            group=[len(train_idx)],
            free_raw_data=False,
        )
        dev = lgb.Dataset(
            X_train.iloc[dev_idx],
            label=y_train.iloc[dev_idx],
            group=[len(dev_idx)],
            free_raw_data=False,
        )
        split_result = {}
        lgb.train(
            parameters,
            train,
            valid_sets=[train, dev, validation],
            valid_names=["train", "dev", "validation"],
            evals_result=split_result,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )
        for data_name, scores in split_result.items():
            for score_name, score_values in scores.items():
                metrics[f"split{f}_{data_name}_{score_name}"] = score_values

    whole_result = {}
    lgb.train(
        parameters,
        whole_train,
        valid_sets=[whole_train, validation],
        valid_names=["train", "validation"],
        evals_result=whole_result,
        num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    for data_name, scores in whole_result.items():
        for score_name, score_values in scores.items():
            metrics[f"whole_{data_name}_{score_name}"] = score_values

    return metrics


def evaluate_predictions(y_true, y_pred, prefix=""):
    metrics = {}
    metrics[prefix + "binary_logloss"] = [log_loss(y_true, y_pred)]
    metrics[prefix + "auc"] = [roc_auc_score(y_true, y_pred)]
    metrics[prefix + "binary_error"] = [
        accuracy_score(y_true, (y_pred > 0.5).astype("int"))
    ]
    return metrics


# TODO too similar to evaluate_lgb_experiment, refactor
def evaluate_logreg_experiment(
    experiment,
    experiment_name,
    n_seeds,
    log_file,
    log_lock,
    X_train,
    X_val,
    y_train,
    y_val,
    folds,
):
    experiment_id, parameters = experiment
    log_data = {}
    log_data["name"] = experiment_name
    log_data["experiment_id"] = experiment_id

    root_seed = parameters["clf__random_state"]
    for sub_seed in range(n_seeds):
        parameters["clf__random_state"] = root_seed + sub_seed
        try:
            log_data.update({"param_" + k: v for k, v in parameters.items()})
            metrics = evaluate_logreg_parameters(
                parameters, X_train, X_val, y_train, y_val, folds
            )
            metrics["success"] = True
            log_data.update(metrics)

        except Exception as e:
            warnings.warn(f'got Exception "{e}" for parameters {parameters}')
            traceback.print_exc(file=sys.stderr)  # TODO use logger instead
        finally:
            with log_lock:
                log_json(log_file, log_data)


def reproduce_logreg_experiment(
    experiment, log_file, log_lock, X_train, X_val, y_train, y_val, folds
):
    name, experiment_id, parameters = experiment
    log_data = {}
    log_data["name"] = name
    log_data["experiment_id"] = experiment_id
    log_data.update({"param_" + k: v for k, v in parameters.items()})
    metrics = evaluate_logreg_parameters(
        parameters,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        folds=folds,
    )
    metrics["success"] = True
    log_data.update(metrics)
    with log_lock:
        log_json(log_file, log_data)


def evaluate_logreg_parameters(
    parameters, X_train, X_val, y_train, y_val, folds
):
    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.values
    continuous = X_train.select_dtypes(include=["number"]).columns.values
    pipe = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", MaxAbsScaler(), continuous),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                    n_jobs=1,
                ),
            ),
            ("clf", LogisticRegression()),
        ]
    )
    parameters = parameters.copy()
    parameters["clf__class_weight"] = {1: parameters["clf__class_weight"]}
    pipe.set_params(**parameters)
    metrics = {}
    for f in range(len(folds)):
        train_idx, dev_idx = folds[f]
        p = clone(pipe)
        p.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        metrics.update(
            evaluate_predictions(
                y_train.iloc[train_idx],
                p.predict_proba(X_train.iloc[train_idx])[:, 1],
                f"split{f}_train_",
            )
        )
        metrics.update(
            evaluate_predictions(
                y_train.iloc[dev_idx],
                p.predict_proba(X_train.iloc[dev_idx])[:, 1],
                f"split{f}_dev_",
            )
        )
        metrics.update(
            evaluate_predictions(
                y_val, p.predict_proba(X_val)[:, 1], f"split{f}_validation_"
            )
        )
    pipe.fit(X_train, y_train)
    metrics.update(
        evaluate_predictions(
            y_train, p.predict_proba(X_train)[:, 1], "whole_train_"
        )
    )
    metrics.update(
        evaluate_predictions(
            y_val, p.predict_proba(X_val)[:, 1], "whole_validation_"
        )
    )
    return metrics


def exclude_columns(df, pattern=None):
    if pattern is not None:
        return df.loc[:, df.columns[~df.columns.str.contains(pattern)]]
    else:
        return df


def summarize_logs(df, n_folds, exclude=None):
    df = (
        df[df.success.fillna(False)]
        .pipe(exclude_columns, pattern=exclude)
        .rename(columns=lambda x: x.replace("@", "_"))
    )

    rows = []
    for row in df.itertuples():
        iterations = pd.DataFrame(
            {
                k: getattr(row, k)
                for k in row._fields
                if k not in ["Index", "param_eval_at", "param_metric"]
            }
        )

        for i, m in product(range(n_folds), METRICS):
            dev = f"split{i}_dev_{m}"
            train = f"split{i}_train_{m}"
            if dev not in iterations.columns or train not in iterations.columns:
                continue
            overfit = f"split{i}_overfit_{m}"
            iterations[overfit] = iterations[train] - iterations[dev]

        for m in SUBSET_METRICS + ["overfit_" + m for m in METRICS]:
            c = [f"split{i}_{m}" for i in range(n_folds)]
            if c[0] not in iterations.columns:
                continue
            iterations["mean_" + m] = iterations[c].mean(axis=1)
            if (
                m.startswith("validation_")
                or m.startswith("dev_")
                or m.startswith("overfit_")
            ):
                iterations["min_" + m] = iterations[c].min(axis=1)
                iterations["max_" + m] = iterations[c].max(axis=1)

        iterations.drop(
            iterations.columns.to_series().filter(regex=r"^split\d+_"),
            axis="columns",
            inplace=True,
        )

        for m in WHOLE_METRICS:
            if m not in iterations.columns:
                continue
            iterations["mean_" + m] = iterations[m]
            iterations["min_" + m] = iterations[m]
            iterations["max_" + m] = iterations[m]
            iterations.drop(m, axis=1, inplace=True)

        if "experiment_id" not in row._fields:
            iterations["experiment_id"] = row.Index

        iterations.index.name = "iteration"
        iterations.reset_index(inplace=True)
        iterations.iteration += 1

        rows.append(optimize_numerics(iterations))

    all_seeds = pd.concat(rows, ignore_index=True)
    all_seeds["cnt"] = 1
    del rows

    aggregations = {}
    for c in all_seeds.columns:
        if c.startswith("mean_"):
            # TODO cast to np.float32
            aggregations[c] = np.mean
        elif c.startswith("min_"):
            aggregations[c] = np.min
        elif c.startswith("max_"):
            aggregations[c] = np.max
        elif c == "cnt":
            aggregations[c] = np.sum
        else:
            aggregations[c] = np.min  # min is faster than lambda with .iloc[0]

    res = (
        all_seeds.groupby(["experiment_id", "iteration"])
        .agg(aggregations)
        .reset_index(drop=True)
    )
    del all_seeds

    return res


# from https://medium.com/bigdatarepublic/advanced-pandas-optimize-speed-and-memory-a654b53be6c2
def optimize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def unfold_iterations(df, n_folds, exclude=None):
    df = (
        df[df.success.fillna(False)]
        .pipe(exclude_columns, pattern=exclude)
        .rename(columns=lambda x: x.replace("@", "_"))
    )

    splits = list(chain([-1], range(n_folds)))
    rows = []
    for row in df.itertuples():
        for s in splits:
            if s == -1:
                one_split_metrics = {
                    m: m.replace("whole_", "", 1) for m in WHOLE_METRICS
                }
            else:
                one_split_metrics = {
                    "split" + str(s) + "_" + m: m for m in SUBSET_METRICS
                }

            iterations = pd.DataFrame(
                {
                    one_split_metrics.get(k): getattr(row, k)
                    for k in row._fields
                    if k in one_split_metrics
                },
                dtype=np.float32,
            )

            iterations["experiment_id"] = row.experiment_id
            if "param_seed" in row._fields:
                iterations["param_seed"] = row.param_seed
            if "param_clf__random_state" in row._fields:
                iterations[
                    "param_clf__random_state"
                ] = row.param_clf__random_state
            iterations["split"] = s

            iterations.index.name = "iteration"
            iterations.reset_index(inplace=True)
            iterations.iteration += 1

            rows.append(optimize_numerics(iterations))

    experiments = (
        df.loc[:, ~df.columns.str.match(r"^(whole_|split|param_seed)")]
        .groupby("experiment_id")
        .first()
        .reset_index()
        .pipe(optimize_numerics)
    )

    return experiments, pd.concat(rows, ignore_index=True, sort=True)


def unfold_to_chunks(
    f, chunk_prefix, n_folds, chunksize=1000, exclude=None, verbose=False
):
    n = 0
    for line in read_json_log(f, chunksize):
        experiments, iterations = unfold_iterations(
            line, n_folds, exclude=exclude
        )
        experiments.to_pickle(f"{chunk_prefix}experiments{n:03d}.pkl")
        iterations.to_pickle(f"{chunk_prefix}iterations{n:03d}.pkl")
        if verbose:
            print(n, "written")
        del experiments, iterations, line
        gc.collect()
        n += 1


def drop_boring_columns(df):
    df.dropna(how="all", axis="columns", inplace=True)
    df.drop(
        columns=["param_eval_at", "param_metric"], errors="ignore", inplace=True
    )
    for c in df.columns:
        if c == "cnt":
            continue
        elif df[c].eq(df[c].iloc[0]).all():
            df.drop(columns=[c], inplace=True)
    return df


def summarize_to_chunks(
    f, chunk_prefix, n_folds, chunksize=1000, exclude=None, verbose=False
):
    n = 0
    for line in read_json_log(f, chunksize):
        chunk = summarize_logs(line, n_folds, exclude=exclude)
        chunk_name = f"{chunk_prefix}{n:03d}.pkl"
        chunk.to_pickle(chunk_name)
        if verbose:
            print(chunk_name, "written")
        del chunk, line
        gc.collect()
        n += 1


def aggregate_chunks(chunks_glob):
    df = pd.concat(
        [pd.read_pickle(f) for f in glob(chunks_glob)],
        ignore_index=True,
        sort=True,
    )
    gc.collect()

    aggregations = {}
    for c in df.columns:
        if c.startswith("mean_"):
            df[c] = df[c] * df.cnt
            aggregations[c] = np.sum
        elif c.startswith("min_"):
            aggregations[c] = np.min
        elif c.startswith("max_"):
            aggregations[c] = np.max
        elif c == "cnt":
            aggregations[c] = np.sum
        else:
            aggregations[c] = np.min  # min is faster than lambda with .iloc[0]
    df = (
        df.groupby(["experiment_id", "iteration"])
        .aggregate(aggregations)
        .pipe(optimize_numerics)
    )
    gc.collect()

    for c in df.columns:
        if c.startswith("mean_"):
            df[c] = df[c] / df.cnt
    gc.collect()

    return drop_boring_columns(df.reset_index(drop=True))


def read_full_logs(f, chunksize=1000, n_chunks=None, exclude=None):
    logs = read_json_log(f, chunksize)
    if n_chunks is not None:
        logs = islice(logs, n_chunks)
    tidy = map(partial(unfold_iterations, exclude=exclude), logs)
    return drop_boring_columns(
        pd.concat(list(tidy), ignore_index=True, sort=True)
    )


def check_omitted_parameters(df):
    all_parameters = (
        df.filter(like="param_", axis="columns")
        .drop("param_seed", axis="columns")
        .pipe(drop_boring_columns)
        .columns.values
    )

    return set(all_parameters) - (
        set(CONT_PARAMETERS)
        | set(LOG_PARAMETERS)
        | set(SET_PARAMETERS)
        | set(INT_PARAMETERS)
    )


def shaderdots(
    df,
    x,
    y,
    plot_width,
    plot_height,
    category_column=None,
    x_axis_type="linear",
):
    def image_callback(x_range, y_range, w, h):
        if category_column:
            agg = ds.count_cat(category_column)
        else:
            agg = ds.count()
        return tf.dynspread(
            tf.shade(
                ds.Canvas(
                    plot_width=w,
                    plot_height=h,
                    x_range=x_range,
                    y_range=y_range,
                    x_axis_type=x_axis_type,
                ).points(df, x, y, agg),
            ),
            max_px=1,
            threshold=0.5,
        )

    p = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        x_axis_type=x_axis_type,
        x_range=df[x].agg(["min", "max"]).values,
        y_range=df[y].agg(["min", "max"]).values,
    )
    p.xaxis[0].axis_label = x
    p.yaxis[0].axis_label = y

    return InteractiveImage(p, image_callback)


def read_files(files, query=None):
    for f in files:
        df = pd.read_pickle(f).assign(file=f)
        if query is None:
            yield df
        else:
            yield df.query(query)


def top_mean_dev_auc(dfs, n):
    return (
        pd.concat(
            list(
                map(
                    lambda df: df.sort_values(
                        "mean_dev_auc", ascending=False
                    ).iloc[:n],
                    dfs,
                )
            ),
            ignore_index=True,
            sort=True,
        )
        .sort_values("mean_dev_auc", ascending=False)
        .iloc[:n]
    )


def top_min_dev_auc(dfs, n):
    return (
        pd.concat(
            list(
                map(
                    lambda df: df.sort_values(
                        "min_dev_auc", ascending=False
                    ).iloc[:n],
                    dfs,
                )
            ),
            ignore_index=True,
            sort=True,
        )
        .sort_values("min_dev_auc", ascending=False)
        .iloc[:n]
    )


def top_min_whole_validation_auc(dfs, n):
    return (
        pd.concat(
            list(
                map(
                    lambda df: df.sort_values(
                        "min_whole_validation_auc", ascending=False
                    ).iloc[:n],
                    dfs,
                )
            ),
            ignore_index=True,
            sort=True,
        )
        .sort_values("min_whole_validation_auc", ascending=False)
        .iloc[:n]
    )


def pre_compare_log(log, n_folds):
    experiments, iterations = unfold_iterations(read_json_log(log), n_folds)
    if "param_seed" in experiments.columns:
        sort_experiments = ["experiment_id", "param_seed"]
        sort_iterations = ["experiment_id", "param_seed", "split", "iteration"]
    elif "param_clf__random_state" in experiments.columns:
        sort_experiments = ["experiment_id", "param_clf__random_state"]
        sort_iterations = [
            "experiment_id",
            "param_clf__random_state",
            "split",
            "iteration",
        ]
    else:
        sort_experiments = ["experiment_id"]
        sort_iterations = ["experiment_id", "split", "iteration"]
    experiments = (
        experiments.drop(columns="timestamp")
        .sort_values(sort_experiments)
        .reset_index(drop=True)
    )
    iterations = iterations.sort_values(sort_iterations).reset_index(drop=True)
    return experiments, iterations


def assert_logs_equal(left_log, right_log, n_folds):
    left_experiments, left_iterations = pre_compare_log(left_log, n_folds)
    right_experiments, right_iterations = pre_compare_log(right_log, n_folds)
    assert_frame_equal(left_experiments, right_experiments, check_exact=True)
    assert_frame_equal(left_iterations, right_iterations)


def quantile_bins(df, x, y, quantiles, bins, quantile_split):
    if quantile_split:
        cut, edges = pd.qcut(df[x], bins, retbins=True)
    else:
        cut, edges = pd.cut(df[x], bins, retbins=True)
    cut.cat.categories = edges[:-1]
    aggs = [(str(q), partial(pd.Series.quantile, q=q)) for q in quantiles]
    aggregated = df.groupby(cut)[y].agg(aggs)
    aggregated.index = aggregated.index.astype(float)
    return aggregated


def experiment_quantiles(
    experiments, folds, x, y, quantiles, bins, quantile_split
):
    return quantile_bins(
        pd.merge(
            folds[["experiment_id", y]],
            experiments[["experiment_id", x]],
            copy=False,
        ).reset_index(drop=True),
        x,
        y,
        quantiles,
        bins,
        quantile_split,
    )
