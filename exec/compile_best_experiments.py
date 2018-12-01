''' compiles the best model for each state into a single experiment result file.
'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
from lib import config
from lib.analysis import scoring


# input file name
TAG = 'ath'

best_models = {}

for state in config.STATES:
    try:
        print state

        ath_models = pd.read_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(state, TAG))
        ath_ranks = pd.read_csv(config.STATES_DIR + '/{0}/{1}_table.csv'.format(state, TAG))

        ### isolate best performing model, excluding GFT or AR52 benchmarks ###

        # count total number of models in file
        assert ath_ranks.SCORE.values[0] == 0, "First row should be a metric with score of 0"
        n_models = 0
        while True:
            if ath_ranks.SCORE.values[n_models + 1] == 0:
                break
            n_models += 1

        # select Pearson correlation scores, excluding benchmarks
        tmp_models = ath_ranks.iloc[1:n_models + 1]
        tmp_models = tmp_models[~tmp_models.Metric.isin(['GFT','AR52'])]

        # select top model
        ath_top_model = tmp_models.loc[tmp_models.SCORE.idxmax()].Metric
        ath_predictions = ath_models[ath_top_model]

        # record best model
        best_models[state] = ath_top_model

        # compile new prediction file
        week = ath_models['Week']
        target = ath_models['ILI']
        gft = ath_models['GFT']
        ar = ath_models['AR52']
        final = pd.concat([week, target, gft, ar, ath_predictions], axis=1)
        final.columns = ['Week','ILI','GFT','AR52','ARGO']

        # save to file
        outfile = config.STATES_DIR + '/{0}/top_argo_preds.csv'.format(state)
        outtable = config.STATES_DIR + '/{0}/top_argo_table.csv'.format(state)
        final.to_csv(outfile, index=False)

        # regenerate score table
        A = scoring.Scorer(state, show_terminal=True, gft_window=True)
        A.results_load(outfile)
        A.results_score()
        A.results_save(outtable)

    except Exception as e:
        print e

with open(config.STATES_DIR + '/_overview/best_model_dict.txt', 'w') as f:
        json.dump(best_models, f, indent=4)
