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
from bokeh.resources import Resources

from lightgbm_tuning import LOGREG_LOG_PARAMETERS, LOGREG_CONT_PARAMETERS, shaderdots

output_notebook(resources=Resources(mode='inline'))
# -

rcParams['figure.figsize'] = 20, 5
pd.set_option('display.max_columns', None)

# %load_ext autoreload
# %autoreload 2

df = pd.concat([pd.read_pickle(f) 
                  for f in ['./experiments/logreg.pkl',
                            './experiments/logreg-liblinear.pkl'
                           ]], 
               ignore_index=True, sort=True)

df.mean_dev_auc.describe()

df.min_whole_validation_auc.describe()

display(shaderdots(df, 'mean_dev_auc', 'min_whole_validation_auc', 700, 700))

display(shaderdots(df, 'mean_train_auc', 'mean_validation_auc', 700, 700))

[display(shaderdots(df, p, 'mean_dev_auc', 500, 500, x_axis_type='log'))
 for p in sorted(set(LOGREG_LOG_PARAMETERS) & set(df.columns))];

[display(shaderdots(df, p, 'mean_dev_auc', 500, 500))
 for p in sorted(set(LOGREG_CONT_PARAMETERS) & set(df.columns))];

# ### overfit auc

df.max_overfit_auc.describe()

df.mean_overfit_auc.describe()

[display(shaderdots(df, p, 'max_overfit_auc', 500, 500, x_axis_type='log'))
 for p in sorted(set(LOGREG_LOG_PARAMETERS) & set(df.columns))];

[display(shaderdots(df, p, 'max_overfit_auc', 500, 500))
 for p in sorted(set(LOGREG_CONT_PARAMETERS) & set(df.columns))];

display(shaderdots(df, 'mean_dev_auc', 'max_overfit_auc', 500, 500))

display(shaderdots(df, 'min_dev_auc', 'max_overfit_auc', 500, 500))

# ### range auc

df['range_auc'] = df.max_dev_auc - df.min_dev_auc

display(shaderdots(df, 'range_auc', 'max_overfit_auc', 500, 500))

[display(shaderdots(df, p, 'range_auc', 500, 500, x_axis_type='log'))
 for p in sorted(set(LOGREG_LOG_PARAMETERS) & set(df.columns))];

[display(shaderdots(df, p, 'range_auc', 500, 500))
 for p in sorted(set(LOGREG_CONT_PARAMETERS) & set(df.columns))];

# ### whole vs cv training

df['whole_cv_diff_min_auc'] = df.min_whole_validation_auc - df.min_validation_auc
df['whole_cv_diff_mean_auc'] = df.mean_whole_validation_auc - df.mean_validation_auc

df[['whole_cv_diff_min_auc', 'whole_cv_diff_mean_auc']][df.mean_train_auc > 0.65].hist(bins=500);

df[['whole_cv_diff_min_auc', 'whole_cv_diff_mean_auc']][df.mean_train_auc > 0.65].describe()
