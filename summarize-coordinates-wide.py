from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=20)
chunked_apply('experiments/wide-best-coordinates.log', 'experiments/wide-best-coordinates-', summarize, verbose=True)
df = aggregate_chunks('experiments/wide-best-coordinates-???.pkl')
df.to_pickle('experiments/wide-best-coordinates.pkl')
