from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=5)
chunked_apply('experiments/good.log.xz', 'experiments/good-overfit-', 5, verbose=True)
df = aggregate_chunks('experiments/good-overfit-???.pkl')
df.to_pickle('experiments/good-overfit.pkl')
