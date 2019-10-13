from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=20)
chunked_apply('experiments/wide-20shuffle-3seed.log.xz', 'experiments/wide-20shuffle-3seed-', summarize, verbose=True)
df = aggregate_chunks('experiments/wide-20shuffle-3seed-???.pkl')
df.to_pickle('experiments/wide-20shuffle-3seed.pkl')
