'''
*The Net and ensemble models were originally implemented in R and are here re-implemented in Python. Due to implementation differences in the lasso routine between R's glmnet and Python's sklearn and some randomness in variable selection during optimization, metrics can vary between runs. However, the overall results are consistent.*
'''


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

from lib import config


def progress_indicator(n):
    sys.stdout.write(str(n))
    sys.stdout.write('\r')
    sys.stdout.flush()


def main():

    VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                    'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                    'SD','TN','TX','UT','VA','VT','WA','WI','WV']

    PRED_START = '2012-09-30'
    PRED_END = '2017-05-14'


    # compile ILI dataframe starting from 2009-10-04
    ili_list = []
    for i, state in enumerate(VALID_STATES):

        state_dat = pd.read_csv(config.STATES_DIR + '/{0}/{0}_merged_data.csv'.format(state),
                                parse_dates=[0])
        ili = state_dat[state]
        ili_list.append(ili)

        if i == 0:
            time_index = pd.to_datetime(state_dat['Week'])

    ili_df = pd.concat([x for x in ili_list], axis=1)
    ili_df.index = time_index


    # compile argo prediction dataframe starting from 2012-09-30
    argo_list = []
    for i, state in enumerate(VALID_STATES):

        state_dat = pd.read_csv(config.STATES_DIR + '/{0}/top_argo_preds.csv'.format(state),
                                parse_dates=[0])
        state_dat = state_dat.set_index(pd.to_datetime(state_dat['Week']))
        state_dat = state_dat[PRED_START:]
        state_dat[state] = state_dat['ARGO']

        argo = state_dat[state]
        argo_list.append(argo)

        if i == 0:
            time_index = pd.to_datetime(state_dat.index)

    argo_df = pd.concat([x for x in argo_list], axis=1)
    argo_df.index = time_index


    # generate Net predictions
    pred_dates = pd.date_range(start=PRED_START, end=PRED_END, freq='7D')
    states = argo_df.columns.values
    pred_df = pd.DataFrame(index=pred_dates, columns=states)
    pred_df = pred_df.fillna(0)

    for st in states:
        print st

        #### generate prediction dataframe ####
        # construct in-state autoregressive matrix
        instate_ili = ili_df[st]
        ar_list = [instate_ili.shift(i) for i in range(53)]
        y = pd.concat(ar_list, axis=1)
        y.columns = ['y_t'] + ['AR{0}'.format(x) for x in range(1, 53)]

        # construct out-of-state network matrix
        out_states = states[states != st]
        outstate_ili = ili_df[out_states]
        out_ar_list = [outstate_ili.shift(i) for i in range(4)]
        column_names = [x + '-{0}'.format(i) for i in range(4) for x in out_states]
        ys = pd.concat(out_ar_list, axis=1)
        ys.columns = column_names

        # concatenate matrices
        y_df = pd.concat([y, ys], axis=1)

        # split predictors and target variables
        target = y_df['y_t']
        design_mat = y_df.drop('y_t', axis=1)

        # generate each week's prediction
        print 'Generating predictions:'
        state_preds = []
        for i, wk in enumerate(pred_dates):
            progress_indicator(wk)

            # set training period window
            train_start = wk - pd.Timedelta(104*7, unit='D')
            train_end = wk - pd.Timedelta(7, unit='D')

            # generate training input
            ys_t = design_mat[design_mat.index==wk]
            ys_t_hat = argo_df.loc[argo_df.index==wk, out_states]

            # replace ARGO predictions as ys_t surrogate values
            col2 = [x + '-0' for x in out_states]
            ys_t[col2] = ys_t_hat[out_states]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(design_mat[train_start:train_end])
            X_test = scaler.transform(ys_t)

            # fit lasso model on training set
            kf = KFold(X_train.shape[0], n_folds=10, shuffle=True)
            lr = LassoCV(normalize=False, cv=kf, max_iter=5000)
            lr.fit(X_train, target[train_start:train_end])

            pred_df.loc[pred_df.index==wk, st] = lr.predict(X_test)[0]
            print lr.predict(X_test)[0]

    pred_df.to_csv(config.STATES_DIR + '/_overview/Predictions_Net.csv')


if __name__ == '__main__':
    main()
