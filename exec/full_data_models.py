''' Runs suite of GT+athena models '''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd

from lib import config
from lib.analysis import scoring
from lib.state_class import One_State


STATES = config.STATES


def main():

    # loop over all states
    for ST in STATES:
        try:
            # initialize state class
            state = One_State(ST, pred_start_date='2012-09-30', verbose=False)

            # clean input data so that all nans are 0
            # (note: calling state.target and state.inputs is a hack into the One_State object)
            state.target[np.isnan(state.target)] = 0
            for key in state.inputs:
                state.inputs[key][np.isnan(state.inputs[key])] = 0

            # run all models

            ar = state.init_predictor(label='AR52')
            ar.data_process(AR_terms=52)
            ar.predict(input_vars=['ar'], method='lasso', horizon=0)

            argo = state.init_predictor(label='ARGO')
            argo.data_process(AR_terms=52)
            argo.predict(input_vars=['ar', 'ath', 'gt'], method='lasso', horizon=0)

            p1 = state.init_predictor(label='ARGO_top10')
            p1.data_process(AR_terms=52)
            p1.predict(input_vars=['ar', 'ath', 'gt'], method='lasso', horizon=0, k_best=10)


            p2 = state.init_predictor(label='ARGO_all_weighted1')
            p2.data_process(AR_terms=52)
            p2.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso', horizon=0, k_best=150)

            p3 = state.init_predictor(label='ARGO_all_weighted2')
            p3.data_process(AR_terms=52)
            p3.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso2', horizon=0, k_best=150)

            p4 = state.init_predictor(label='ARGO_all_weighted_more_ar')
            p4.data_process(AR_terms=52)
            p4.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso3', horizon=0, k_best=150)


            p5 = state.init_predictor(label='ARGO_top10_weighted1')
            p5.data_process(AR_terms=52)
            p5.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso', horizon=0, k_best=10)

            p6 = state.init_predictor(label='ARGO_top10_weighted2')
            p6.data_process(AR_terms=52)
            p6.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso2', horizon=0, k_best=10)

            p7 = state.init_predictor(label='ARGO_top10_weighted_more_ar')
            p7.data_process(AR_terms=52)
            p7.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso3', horizon=0, k_best=10)


            # p8 = state.init_predictor(label='ARGO_ath_weight')
            # p8.data_process(AR_terms=52)
            # p8.predict(input_vars=['ar', 'ath', 'gt'], method='ath-weight', horizon=0)
            #
            # p9 = state.init_predictor(label='ARGO_ath_deweight')
            # p9.data_process(AR_terms=52)
            # p9.predict(input_vars=['ar', 'ath', 'gt'], method='ath-deweight', horizon=0)


            p10 = state.init_predictor(label='ARGO_obs_weight')
            p10.data_process(AR_terms=52)
            p10.predict(input_vars=['ar', 'ath', 'gt'], method='lasso-obs', horizon=0)

            p11 = state.init_predictor(label='ARGO_top10_obs')
            p11.data_process(AR_terms=52)
            p11.predict(input_vars=['ar', 'ath', 'gt'], method='lasso-obs', horizon=0, k_best=10)

            p12 = state.init_predictor(label='ARGO_top10_weighted2_obs')
            p12.data_process(AR_terms=52)
            p12.predict(input_vars=['ar', 'ath', 'gt'], method='weighted-lasso2-obs', horizon=0, k_best=10)


            # save predictions to file
            outfile = config.STATES_DIR + '/{0}/ath_preds.csv'.format(ST)
            outtable = config.STATES_DIR + '/{0}/ath_table.csv'.format(ST)
            state.save_to_csv(outfile)

            # read in GFT and save as new column to outfile without modification
            saved_df = pd.read_csv(outfile)
            gft = pd.read_csv(config.STATES_DIR + '/{0}/GFT_scaled.csv'.format(ST), parse_dates=[0])
            gft_vals = gft.GFT.values
            saved_df['GFT'] = gft_vals
            saved_df.to_csv(outfile, index=False)

            # score prediction file and save table to file
            A = scoring.Scorer(ST, show_terminal=True, gft_window=True)
            A.results_load(outfile)
            A.results_score()
            A.results_save(outtable)

        except Exception as e:
            print e


if __name__ == '__main__':

    main()
