import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold

from lib import config
from compile_prediction_dataframe import compile


VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']


def rmse(x1, x2):
    return np.sqrt(np.mean((x1 - x2) ** 2))


def predict_ensemble(y, p1, p2, K):
    p = np.zeros(len(y))
    for i in range(K, len(y)):
        if rmse(y[i-K:i-1], p1[i-K:i-1]) < rmse(y[i-K:i-1], p2[i-K:i-1]):
            p[i] = p1[i]
        else:
            p[i] = p2[i]
    return p


def main():
    TAG = 'top_argo_preds'
    OUTFILE = 'compiled_argo_preds'
    df = compile(TAG, OUTFILE)

    argo = pd.read_csv(config.STATES_DIR + '/_overview/compiled_argo_preds.csv', index_col=0,
                       infer_datetime_format=True)
    net = pd.read_csv(config.STATES_DIR + '/_overview/Predictions_Net.csv', index_col=0,
                      infer_datetime_format=True)
    Ks = pd.read_csv(config.DATA_DIR + '/analysis/state_K.csv', index_col=0)

    pred_dates = pd.date_range(start='2012-09-30', end='2017-05-14', freq='7D')
    pred_df = pd.DataFrame(index=pred_dates, columns=VALID_STATES)
    pred_df = pred_df.fillna(0)

    for st in VALID_STATES:
        p1 = argo[argo.state == st]
        p1 = p1['2012-09-30':]

        y = p1.ILI
        p1 = p1.ARGO.values
        p2 = net[st].values

        K = Ks.loc[st, 'K']

        pred = predict_ensemble(y, p1, p2, K)
        pred_df.loc[:, st] = pred

    pred_df = pred_df['2014-09-28':]
    pred_df.index.names = ['Week']
    pred_df.to_csv(config.STATES_DIR + '/_overview/Predictions_Ensemble.csv')

if __name__ == '__main__':
    main()
