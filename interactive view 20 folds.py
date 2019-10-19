# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import glob
import gc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

from pylab import rcParams
from IPython.display import display

from utils import METRICS, CONT_PARAMETERS, LOG_PARAMETERS, SET_PARAMETERS, INT_PARAMETERS, shaderdots,\
    quantile_bins, experiment_quantiles

from bokeh.io import output_notebook

output_notebook()
# -

rcParams['figure.figsize'] = 20, 5
pd.set_option('display.max_columns', None)

# %load_ext autoreload
# %autoreload 2

n_chunks = 10

experiments = pd.concat(
    [pd.read_pickle(f'experiments/wide-20shuffle-3seed-fullexperiments{n:03d}.pkl') for n in range(n_chunks)],
    ignore_index=True,
    sort=True)\
.groupby('experiment_id').first().reset_index()

folds = pd.concat(
    [pd.read_pickle(f'experiments/wide-20shuffle-3seed-fulliterations{n:03d}.pkl') for n in range(n_chunks)],
    ignore_index=True,
    sort=True)

# +
#experiments.to_pickle('experiments/wide-20shuffle-3seed-full-experiments.pkl')
#folds.to_pickle('experiments/wide-20shuffle-3seed-full-iterations.pkl')

#experiments = pd.read_pickle('experiments/wide-20shuffle-3seed-full-experiments.pkl')
#folds = pd.read_pickle('experiments/wide-20shuffle-3seed-full-iterations.pkl')
# -

folds['split'] = pd.Categorical(folds.split)
cv_folds = folds.query('split != -1').reset_index(drop=True)
whole_folds = folds.query('split == -1').reset_index(drop=True)
del folds
gc.collect()
cv_folds['overfit_auc'] = cv_folds.train_auc - cv_folds.dev_auc

quantiles = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1]
bins = 75

[experiment_quantiles(experiments, cv_folds, p, 'dev_auc', quantiles, bins, quantile_split=True)\
        .plot(logx=True, legend=False, grid=True)
    for p in sorted(set(LOG_PARAMETERS) & set(experiments.columns))];

[experiment_quantiles(experiments, cv_folds, p, 'dev_auc', quantiles, bins, quantile_split=False)\
        .plot(logx=True, legend=False, grid=True)
    for p in sorted((set(CONT_PARAMETERS) | set(INT_PARAMETERS)) & set(experiments.columns))];

[experiment_quantiles(experiments, cv_folds, p, 'overfit_auc', quantiles, bins, quantile_split=True)\
        .plot(logx=True, legend=False, grid=True)
    for p in sorted(set(LOG_PARAMETERS) & set(experiments.columns))];

[experiment_quantiles(experiments, cv_folds, p, 'overfit_auc', quantiles, bins, quantile_split=False)\
        .plot(logx=True, legend=False, grid=True)
    for p in sorted((set(CONT_PARAMETERS) | set(INT_PARAMETERS)) & set(experiments.columns))];

display(shaderdots(cv_folds, 'dev_auc', 'validation_auc', 700, 700, category_column='split'))
