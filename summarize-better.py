from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=20)
chunked_apply('experiments/better.log', 'experiments/better-', summarize, verbose=True)
df = aggregate_chunks('experiments/better-???.pkl')
df.to_pickle('experiments/better.pkl')
