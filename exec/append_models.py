'''  Performs 3 utilities:

1. When running a suite of models in full_data_models.py in separate batches, different
but similar files are made with model predictions.

The function main combines these into a single prediction file for each state
(identical to if the entire suite were run at once).

2. The function ensemble_append adds a column to the above prediction files with ensemble
predictions, which are generated in a single file in the results/_overview directory.

3. The function net_append does the same for net predictions.

'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from lib import config
from lib.analysis import scoring
from lib.state_class import One_State

STATES = config.STATES
FILE1 = 'top_net'
FILE2 = ''
OUTFILE = 'top_ens'

def main():

    for st in config.STATES:
        try:
            print st

            home_models = pd.read_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, FILE1))
            add_models = pd.read_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, FILE2))

            merged = home_models.merge(add_models)

            merged.to_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, OUTFILE), index=False)


            A = scoring.Scorer(st, show_terminal=True, gft_window=True)
            A.results_load(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, OUTFILE))
            A.results_score()
            A.results_save(config.STATES_DIR + '/{0}/{1}_table.csv'.format(st, OUTFILE))

        except Exception as e:
            print e


def ensemble_append(FILE1, OUTFILE):

    ens = pd.read_csv(config.STATES_DIR + '/_overview/Predictions_Ensemble.csv', parse_dates=[0])

    for st in config.STATES:
        try:
            print st

            home_models = pd.read_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, FILE1), parse_dates=[0])
            add_models = ens.loc[:, ['Week', st]]
            merged = home_models.merge(add_models, on='Week')
            merged.rename(columns={st:'ARGONet'}, inplace=True)

            merged.to_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, OUTFILE), index=False)

            A = scoring.Scorer(st, show_terminal=True, gft_window=True)
            A.results_load(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, OUTFILE))
            A.results_score()
            A.results_save(config.STATES_DIR + '/{0}/{1}_table.csv'.format(st, OUTFILE))

        except Exception as e:
            print e


def net_append(FILE1, OUTFILE):

    net = pd.read_csv(config.STATES_DIR + '/_overview/Predictions_Net.csv', parse_dates=[0])

    for st in config.STATES:
        try:
            print st

            home_models = pd.read_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, FILE1), parse_dates=[0])
            add_models = net.loc[:, ['Week', st]]
            merged = home_models.merge(add_models, on='Week')

            merged.rename(columns={st:'Net'}, inplace=True)

            merged.to_csv(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, OUTFILE), index=False)


            A = scoring.Scorer(st, show_terminal=True, gft_window=True)
            A.results_load(config.STATES_DIR + '/{0}/{1}_preds.csv'.format(st, OUTFILE))
            A.results_score()
            A.results_save(config.STATES_DIR + '/{0}/{1}_table.csv'.format(st, OUTFILE))

        except Exception as e:
            print e


if __name__ == '__main__':

    FILE1 = 'top_argo'
    OUTFILE = 'argo_net'
    net_append(FILE1, OUTFILE)

    FILE1 = 'argo_net'
    OUTFILE = 'argo_net_ens'
    ensemble_append(FILE1, OUTFILE)
