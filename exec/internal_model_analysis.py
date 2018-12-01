''' Compiles a heatmap of the frequency each specific ARGO modification turns out to be the
best-performing version over all the states.

This code was used for sanity checking, and was not used to generate any results or figures
in the paper.

'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from lib import config


for st in ['AL']:

    results = pd.read_csv(config.STATES_DIR + '/{0}/gt_table.csv'.format(st))
    start = results[results.Metric == 'PEARSON'].index[0]
    end = results[results.Metric == 'RMSE'].index[0]

    n_models = end - start - 1

    model_names = results.Metric.values[start + 1:end]

    lookup = {}
    for i in range(n_models):
        lookup[model_names[i]] = i
    print lookup


ranks = np.zeros((n_models, n_models))

for st in config.STATES:
    try:
        results = pd.read_csv(config.STATES_DIR + '/{0}/gt_table.csv'.format(st))
        start = results[results.Metric == 'PEARSON'].index[0]
        end = results[results.Metric == 'RMSE'].index[0]

        models = results.Metric.values[start + 1:end]
        scores = results.SCORE.values[start + 1:end]
        order = np.argsort(scores)[::-1]

        for i in range(len(order)):
            ith_rank_model = models[order[i]]
            true_index = lookup[ith_rank_model]
            ranks[i, true_index] += 1

    except Exception as e:
        print e

# print ranks

final_ranks = pd.DataFrame(ranks, columns=model_names)

fig = sns.heatmap(final_ranks, cmap='Reds', yticklabels=np.arange(1, len(model_names) + 1), annot=True)
plt.title('Model rank frequencies')
for item in fig.get_xticklabels():
    item.set_rotation(90)
# plt.xticks(np.arange(len(model_names)), model_names, rotation='vertical')
# axes[1,0].set_yticklabels(['0','1','2','3'], rotation='horizontal')
# axes[1,0].set_ylabel('Seasons outperforming GFT')
plt.gcf().subplots_adjust(bottom=0.24)
plt.show()
