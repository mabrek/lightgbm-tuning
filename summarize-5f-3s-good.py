from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=5)
chunked_apply('experiments/good-5fold-3seed.log.xz', 'experiments/good-5fold-3seed-', summarize, verbose=True)
df = aggregate_chunks('experiments/good-5fold-3seed-???.pkl')
df.to_pickle('experiments/good-5fold-3seed.pkl')
