from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=20)
chunked_apply('experiments/better-best-coordinates.log', 'experiments/better-best-coordinates-', summarize, verbose=True)
df = aggregate_chunks('experiments/better-best-coordinates-???.pkl')
df.to_pickle('experiments/better-best-coordinates.pkl')
