#!/usr/bin/env python

from multiprocessing import Lock
import argparse

from utils import read_json_log, read_telecom_churn, run_pool,\
    evaluate_logreg_parameters, log_json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Reproduce recorded parameters')
    parser.add_argument('--input-log', required=True)
    parser.add_argument('--output-log', required=True)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--chunksize', type=int, default=10)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--split-kind',
                        type=str,
                        default='k-folds',
                        choices=['k-folds', 'shuffle-split'])
    args = parser.parse_args()

    log_lock = Lock()
    X_train, X_val, y_train, y_val, folds = \
        read_telecom_churn(args.n_folds, args.split_kind)

    input_log = read_json_log(args.input_log)

    def generator():
        for _, row in input_log.iterrows():
            parameters = row.filter(regex='^param_')\
                            .rename(lambda x: x.replace('param_', ''))\
                            .to_dict()
            yield (row['name'],
                   row['experiment_id'],
                   parameters)

    def evaluator(experiment):
        name, experiment_id, parameters = experiment
        log_data = {}
        log_data['name'] = name
        log_data['experiment_id'] = experiment_id
        log_data.update({'param_' + k: v for k, v in parameters.items()})
        metrics = evaluate_logreg_parameters(
            parameters,
            X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
            folds=folds)
        metrics['success'] = True
        log_data.update(metrics)
        with log_lock:
            log_json(args.output_log, log_data)

    run_pool(generator(), args, evaluator)
