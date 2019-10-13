from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=5)
chunked_apply('experiments/wide-5fold-3seed.log', 'experiments/wide-5fold-3seed-', summarize, verbose=True)
df = aggregate_chunks('experiments/wide-5fold-3seed-???.pkl')
df.to_pickle('experiments/wide-5fold-3seed.pkl')
