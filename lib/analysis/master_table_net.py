import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                'SD','TN','TX','UT','VA','VT','WA','WI','WV']

HEADER = ['2012-13', '2013-14', '2014-15', '2015-16', '2016-17', 'Whole Period', 'GFT Period']
METRICS = ['RMSE', 'PEARSON', 'MAPE']
MODELS = ['GFT', 'AR52', 'ARGO(gt)', 'ARGO(gt,ath)', 'Net']


class Tabler(object):
    def __init__(self, VALID_STATES, HEADER, METRICS, MODELS):
        self.states = VALID_STATES
        self.header = HEADER
        self.metrics = METRICS
        self.models = MODELS
        self.index = None
        self.complete = None

    def multi_index(self):

        models_expanded = np.tile(np.tile(self.models, len(self.states)), len(self.metrics))
        states_expanded = np.tile(np.repeat(self.states, len(self.models)), len(self.metrics))
        metrics_expanded = np.repeat(self.metrics, len(self.states) * len(self.models))

        tuples = list(zip(metrics_expanded, states_expanded, models_expanded))

        index = pd.MultiIndex.from_tuples(tuples, names=['Metric', 'State', 'Model'])

        self.index = index


    def load_tables(self):

        all_rows = [[] for x in range(len(self.metrics))]

        for ST in self.states:
            results = pd.read_csv(config.STATES_DIR + '/{0}/top_net_table.csv'.format(ST))
            results = results.drop(labels='SCORE', axis=1)

            dividers = []
            for key in self.metrics:
                dividers.append(results[results.Metric == key].index.tolist()[0])

            for i in range(len(dividers)):

                results_one_metric = results[dividers[i] + 1:dividers[i] + 1 + len(self.models)]

                gft_row = results_one_metric[results_one_metric.Metric == 'GFT'].values[:, 1:]
                ar_row = results_one_metric[results_one_metric.Metric == 'AR52'].values[:, 1:]
                argo1_row = results_one_metric[results_one_metric.Metric == 'ARGO(gt)'].values[:, 1:]
                argo2_row = results_one_metric[results_one_metric.Metric == 'ARGO(gt,ath)'].values[:, 1:]
                argo3_row = results_one_metric[results_one_metric.Metric == 'Net'].values[:, 1:]

                block = np.vstack((gft_row, ar_row, argo1_row, argo2_row, argo3_row))

                all_rows[i].append(block)


        #complete = [np.vstack([x for x in sublist]) for sublist in all_rows]
        complete = np.vstack([item for sublist in all_rows for item in sublist])

        self.complete = complete


    def finish_table(self):
        final = pd.DataFrame(self.complete, index=self.index, columns=self.header)

        final.to_csv(config.STATES_DIR + '/_overview/final_table_net.csv', na_rep='--')
        final.to_excel(config.STATES_DIR + '/_overview/final_table_net.xlsx', na_rep='--')

if __name__ == '__main__':

    a = Tabler(VALID_STATES, HEADER, METRICS, MODELS)
    a.multi_index()
    a.load_tables()
    a.finish_table()
