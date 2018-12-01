'''
scoring.py
Fred Lu
6/13/17

Scoring utility for seasonal flu predictions with flexible parameters.

Takes as input a CSV file with model results in the following format:

    Week      ILI     Model1    Model2    ...
   1/13/13    2.2       2.4       1.1     ...
   1/20/13    3.4       3.3       1.2     ...
     ...      ...       ...       ...     ...


The scoring utility computes performance between each model and the target over
the entire input CSV as well as over specified time periods (e.g. flu seasons),
using a flexible set of metrics.

In order to compute metrics over specific time periods, a lookup table is read
with the following format:

    Sunday    Season    ...
   12/30/12    2012     ...
   1/06/13     2013     ...
   1/13/13     2013     ...

This table is merged with the input file.
The output is a CSV with the following format:

                          Season1    Season2    Season3    ...
    Metric1    Model1        x1         x2         x3      ...
    Metric1    Model2        x4         x5         x6      ...
    Metric2    Model1        x7         x8         x9      ...
    Metric2    Model2        x10        x11        x12     ...
      ...       ...          ...        ...        ...     ...


'''


from __future__ import division
import sys
import numpy as np
from scipy.stats.stats import pearsonr
from math import sqrt
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def filt(x, y):
    mat = np.array(zip(x, y))
    mat = mat[~np.any(mat==0, axis=1) & ~np.any(mat==-.001, axis=1)]
    return mat[:, 0], mat[:, 1]

# scoring metric functions
def RMSE(predictions, targets):
    predictions, targets = filt(predictions, targets)
    return sqrt(((predictions - targets) ** 2).mean())


def PEARSON(predictions, targets):
    predictions, targets = filt(predictions, targets)
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]


def MAE(predictions, targets):
    predictions, targets = filt(predictions, targets)
    return np.absolute(predictions - targets).mean()


def MAPE(predictions, targets):
    predictions, targets = filt(predictions, targets)
    return np.mean(np.abs((predictions - targets)) / targets)


# define which metrics to be used here
METRICS = [PEARSON, RMSE, MAPE]
LOOKUP = config.DATA_DIR + '/ILI/CDC_season_lookup.csv'


class Scorer(object):
    ''' Scorer is a utility that produces scoring metrics on an input state with
        model predictions.

            @Params: ST = 'MA', or other state
    '''

    def __init__(self, ST, show_terminal=False, gft_window=False):
        self.state = ST
        self.methods = None
        self.seasons = None
        self.metrics = METRICS
        self.score_array = None
        self.show_terminal = show_terminal
        self.gft_window = gft_window

    def results_load(self, fname='default'):

        fname = self._default_file_loc() if fname == 'default' else fname

        _results = pd.read_csv(fname, parse_dates=['Week'])
        _season_lookup = pd.read_csv(LOOKUP, parse_dates=['Sunday'], dtype=object)

        # check that proper input formatting is met
        methods = _results.columns.values
        assert methods[0] == 'Week', "First column of results must be Week"
        assert methods[1] == 'ILI', "Second column of results must be ILI"
        self.methods = methods[2:]

        # merge result csv with season information from lookup table
        self.combined = _results.merge(_season_lookup[['Sunday', 'Season', 'is_GFT']], how='inner',
                                       left_on='Week', right_on='Sunday')

        # extract parameters
        self.seasons = self.combined.Season.unique()
        self.seasons = self.seasons[~pd.isnull(self.seasons)]

    def results_score(self):

        # construct score array (1 column for each season + 1 for whole period)
        score_array = np.empty((len(self.metrics) * len(self.methods), len(self.seasons) + 1))

        # score over each season as specified in lookup table
        for i, season in enumerate(self.seasons):
            tmp_df = self.combined[self.combined.Season == season]
            score_array[:, i] = self._metrics_single_season(tmp_df)

        # score over entire period of CSV
        score_array[:, -1] = self._metrics_single_season(self.combined)

        # compute score (average result over the seasons where GFT is available)
        df_mean_score = np.nanmean(score_array[:, 2:5], axis=1).reshape(-1, 1)

        # score GFT window if called for
        if self.gft_window is True:
            df_gft_window = self.combined[self.combined.is_GFT == '1']

            gft_column = self._metrics_single_season(df_gft_window).reshape(-1, 1)

            score_array = np.hstack((score_array, gft_column))

        score_array = np.hstack((score_array, df_mean_score))

        self.score_array = np.around(score_array, decimals=3)

    # def results_save(self, fname='default'):

    #     fname = self._default_out_file_loc() if fname == 'default' else fname

    #     n_metrics = len(self.metrics)
    #     n_methods = len(self.methods)
    #     metric_names = [x.__name__ for x in self.metrics]

    #     # generate row labels for csv
    #     methods_column = np.tile(self.methods, n_metrics).reshape(-1, 1)
    #     metrics_column = np.repeat(metric_names, n_methods).reshape(-1, 1)
    #     score_array_labeled = np.hstack((metrics_column, methods_column, self.score_array))

    #     # generate column labels for csv
    #     header = np.insert(self.seasons, 0, ['', '', 'All'])

    #     # save to file
    #     df_final = pd.DataFrame(score_array_labeled, columns=header.tolist())
    #     df_final.to_csv(fname, index=False)

    #     if self.show_terminal:
    #         print df_final

    def results_save(self, fname='default'):

        fname = self._default_out_file_loc() if fname == 'default' else fname

        n_metrics = len(self.metrics)
        n_methods = len(self.methods)

        # insert rows of zeros to support metric name (in reverse to preserve indices)
        if self.gft_window is True:
            row = np.zeros((1, len(self.seasons) + 3))
        else:
            row = np.zeros((1, len(self.seasons) + 2))
        for i in range(n_metrics - 1, -1, -1):
            self.score_array = np.insert(self.score_array, i * n_methods, row, axis=0)

        # generate row labels for csv
        methods_column = np.tile(self.methods, n_metrics).reshape(-1, 1)
        for i in range(n_metrics - 1, -1, -1):
            methods_column = np.insert(methods_column, i * n_methods, self.metrics[i].__name__,
                                       axis=0)

        score_array_labeled = np.hstack((methods_column, self.score_array))

        # generate column labels for csv
        if self.gft_window is True:
            header = np.append(np.insert(self.seasons, 0, ['Metric']), ['ALL_PERIOD', 'GFT_PERIOD', 'SCORE'])
        else:
            header = np.append(np.insert(self.seasons, 0, ['Metric']), 'ALL_PERIOD', 'SCORE')

        # save to file
        df_final = pd.DataFrame(score_array_labeled, columns=header.tolist())
        df_final.to_csv(fname, index=False)

        if self.show_terminal:
            print df_final

    def _default_file_loc(self):
        return config.OUT_DIR + '/{0}/{0}_results.csv'.format(self.state)

    def _default_out_file_loc(self):
        return config.OUT_DIR + '/{0}/{0}_table.csv'.format(self.state)

    def _metrics_single_season(self, df):

        output = np.empty(len(self.metrics) * len(self.methods))

        for metric_index, metric in enumerate(self.metrics):
            for m_index, m in enumerate(self.methods):
                output[len(self.methods) * metric_index + m_index] = metric(df[m], df['ILI'])

        return output


def terminal_version():

    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print "Usage: python scoring.py STATE filename.csv or python scoring.py -all"
        sys.exit()

    if len(sys.argv) == 3:

        STATE = sys.argv[1]
        in_file = sys.argv[2]
        out_file = '_scores.csv'.join(sys.argv[2].split('.csv'))

        A = Scorer(STATE, show_terminal=True, gft_window=True)
        A.results_load(in_file)
        A.results_score()
        A.results_save(out_file)

    elif len(sys.argv) == 2 and sys.argv[1] == '-all':

        for ST in config.STATES:

            IN_FILE = config.STATES_DIR + '/{0}/{0}_fred_preds.csv'.format(ST)
            OUT_FILE = config.STATES_DIR + '/{0}/{0}_fred_table.csv'.format(ST)

            A = Scorer(ST, show_terminal=False)
            A.results_load(IN_FILE)
            A.results_score()
            A.results_save(OUT_FILE)
    else:
        print "Usage: python scoring.py STATE filename.csv or python scoring.py -all'"


if __name__ == '__main__':

    terminal_version()
