from functools import partial
from utils import *

summarize = partial(summarize_logs, n_folds=20)
chunked_apply('experiments/good-20shuffle-3seed.log', 'experiments/good-20shuffle-3seed-', summarize, verbose=True)
df = aggregate_chunks('experiments/good-20shuffle-3seed-???.pkl')
df.to_pickle('experiments/good-20shuffle-3seed.pkl')
