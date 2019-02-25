# LightGBM Tuning Experiment

The idea was to get a small and not very predictable dataset and try to measure an effect of LightGBM parameter tuning on classification metrics by running a long random search in a large parameter space.

TLDR: [view experiment results in jupyter notebook](https://nbviewer.jupyter.org/github/mabrek/lightgbm-tuning/blob/export/explore%20experiments.ipynb)

## Steps to reproduce

Get data from https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/ by running

    wget -O  ./data/WA_Fn-UseC_-Telco-Customer-Churn.csv https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv

The same dataset is available at https://www.kaggle.com/blastchar/telco-customer-churn

Docker, very large image from https://kaggle.com, but it has everything:

    docker pull gcr.io/kaggle-images/python@sha256:26b111929a0df780f246fbf3db9f57f8f69c944e898735c59fd8581c42f92f1d

Start container (change path to cloned repo):

    docker run -it --rm -v /data/work/sources/lightgbm-tuning:/lightgbm-tuning --net=host --name lightgbm-tuning gcr.io/kaggle-images/python@sha256:26b111929a0df780f246fbf3db9f57f8f69c944e898735c59fd8581c42f92f1d bash

Run experiments (in docker container):

    cd /lightgbm-tuning/
    ./search-telecom-churn.py --name example-2processes --log experiments/example.log --processes 2 --iterations 10

For better performance increase `--chunksize` to 10-100 when running many `--iterations`, set `--processes` to the number of physical CPU cores available.

You can change dataset, preprocessing steps, and parameter search space by copying and modifying `search-telecom-churn.py`.

Dataset-independent functions go to `utils.py`. PRs are welcome!

Then open `explore experiments.ipynb` in `jupyter-notebook`, load your experiment logs and play with results.

There are some example logs provided in the repository, but they had to be downsampled due to github size limits.

The code is licensed under BSD License.

It contains parts from unmerged pull request from scikit-learn which is (c) by the scikit-learn developers.
