from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=20)
chunked_apply('experiments/good-best-coordinates.log.xz', 'experiments/good-best-coordinates-', summarize, verbose=True)
df = aggregate_chunks('experiments/good-best-coordinates-???.pkl')
df.to_pickle('experiments/good-best-coordinates.pkl')
