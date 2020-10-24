from lightgbm_tuning import *

df = summarize_logs(read_json_log("experiments/logreg-liblinear.log.xz"), 20)
df.to_pickle("experiments/logreg-liblinear.pkl")
