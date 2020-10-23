from utils import *

summarize_to_chunks(
    "experiments/good-best-coordinates.log.xz",
    "experiments/good-best-coordinates-",
    20,
    verbose=True,
)
df = aggregate_chunks("experiments/good-best-coordinates-???.pkl")
df.to_pickle("experiments/good-best-coordinates.pkl")
