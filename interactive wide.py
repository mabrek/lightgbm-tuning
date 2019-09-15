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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

from pylab import rcParams
from IPython.display import display

from bokeh.io import output_notebook

from utils import METRICS, CONT_PARAMETERS, LOG_PARAMETERS, SET_PARAMETERS, INT_PARAMETERS, shaderdots

output_notebook()
# -

rcParams['figure.figsize'] = 20, 5
pd.set_option('display.max_columns', None)

# %load_ext autoreload
# %autoreload 2

df = pd.concat([pd.read_pickle(f) 
                  for f in ['./experiments/wide-20shuffle-3seed.pkl'
                           ]], 
               ignore_index=True, sort=True)

df.mean_dev_auc.describe()

df.min_whole_validation_auc.describe()

display(shaderdots(df, 'min_dev_auc', 'min_whole_validation_auc', 700, 700))

display(shaderdots(df, 'mean_dev_auc', 'min_whole_validation_auc', 700, 700))

[display(shaderdots(df, p, 'mean_dev_auc', 500, 500, x_axis_type='log'))
 for p in sorted(set(LOG_PARAMETERS) & set(df.columns))];

[display(shaderdots(df, p, 'mean_dev_auc', 500, 500))
 for p in sorted((set(CONT_PARAMETERS) | set(INT_PARAMETERS)) & set(df.columns))];

# ### overfit auc

df.max_overfit_auc.describe()

df.mean_overfit_auc.describe()

[display(shaderdots(df, p, 'max_overfit_auc', 500, 500, x_axis_type='log'))
 for p in sorted(set(LOG_PARAMETERS) & set(df.columns))];

[display(shaderdots(df, p, 'max_overfit_auc', 500, 500))
 for p in sorted((set(CONT_PARAMETERS) | set(INT_PARAMETERS)) & set(df.columns))];

display(shaderdots(df, 'mean_dev_auc', 'max_overfit_auc', 500, 500))

# ### range auc

df['range_auc'] = df.max_dev_auc - df.min_dev_auc

[display(shaderdots(df, p, 'range_auc', 500, 500, x_axis_type='log'))
 for p in sorted(set(LOG_PARAMETERS) & set(df.columns))];

[display(shaderdots(df, p, 'range_auc', 500, 500))
 for p in sorted((set(CONT_PARAMETERS) | set(INT_PARAMETERS)) & set(df.columns))];

# ### logloss

display(shaderdots(df, 'param_learning_rate', 'max_overfit_binary_logloss', 500, 500, x_axis_type='log'));

display(shaderdots(df, 'iteration', 'max_dev_binary_logloss', 500, 500));

display(shaderdots(df, 'iteration', 'mean_dev_binary_logloss', 500, 500));

# ### whole vs cv training

df['whole_cv_diff_min_auc'] = df.min_whole_validation_auc - df.min_validation_auc
df['whole_cv_diff_mean_auc'] = df.mean_whole_validation_auc - df.mean_validation_auc

df[['whole_cv_diff_min_auc', 'whole_cv_diff_mean_auc']][df.mean_train_auc > 0.65].hist(bins=500);


