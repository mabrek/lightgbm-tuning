from lightgbm_tuning import *

summarize_to_chunks(
    "experiments/good.log.xz", "experiments/good-overfit-", 5, verbose=True
)
df = aggregate_chunks("experiments/good-overfit-???.pkl")
df.to_pickle("experiments/good-overfit.pkl")
