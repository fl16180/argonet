''' state_class is a wrapper for the forecastlib package that allows interface with the specific
format of the state flu project data.
'''

import pandas as pd
import numpy as np
import copy
import datetime

import config
from forecastlib.models import Predictor


class One_State(object):

    def __init__(self, STATE, pred_start_date='2012-09-30', verbose=True):

        print '\n----------------------START-----------------------'
        print datetime.datetime.now()
        print STATE + ' model initialized.'

        self.STATE = STATE
        self.model_list = []
        self.pred_start_date = pred_start_date
        self.verbose = verbose

        _file = config.STATES_DIR + '/{0}/{0}_merged_data.csv'.format(STATE)

        _all_data = pd.read_csv(_file, parse_dates=[0])
        self.datelist = pd.to_datetime(pd.Series(_all_data.ix[:, 'Week']),
                                       infer_datetime_format=True).tolist()

        self.target = _all_data.iloc[:, [1]].values.astype(float).reshape(-1, )

        self.inputs = {}
        self.inputs['ath'] = _all_data.iloc[:, [2, 3, 4]].values.astype(float)
        self.inputs['gt'] = _all_data.iloc[:, 5:].values.astype(float)

        if verbose is True:
            print 'Target loaded. Length: ', len(self.target)
            print 'Dates loaded. Length: ', len(self.datelist)
            print 'Independent variables: '
            for key in self.inputs:
                print '\t{0} loaded. Length: {1}'.format(key, len(self.inputs[key]))

    def init_predictor(self, label):
        ''' creates a single Predictor object and returns it

            Parameters:
                label: Name of the predictor, used as unique identifier, e.g. 'ARGO (t-1)'.
        '''
        model = Predictor(copy.deepcopy(self.inputs), np.copy(self.target), label,
                          self.pred_start_date, copy.copy(self.datelist), resolution='week',
                          verbose=self.verbose)
        self.model_list.append(model)

        return model

    def save_to_csv(self, fname):
        print 'Saving predictions to file. '

        # manually add 0s to match leo's visualization format
        start_pad_dates = pd.date_range(start=pd.datetime(2011, 1, 2),
                                        end=pd.to_datetime(self.pred_start_date) -
                                            pd.to_timedelta(1, unit='d'),
                                        freq='7D').tolist()
        start_pad_zeros = np.zeros(len(start_pad_dates))


        # create list of models matching model_labels, or if 'all' include all models
        s = [m for m in self.model_list]

        # index all model variables so that they start with user-specified prediction start date
        preds = []
        for model in s:
            shift = model.datelist.index(pd.to_datetime(self.pred_start_date))
            model.datelist = model.datelist[shift:]
            model.target = model.target[shift:]
            model.predictions = model.predictions[shift:]
            tmp = np.concatenate((start_pad_zeros, model.predictions))
            preds.append(pd.Series(tmp, name=model.label))

        # extract target time series from the first model
        tmp = np.concatenate((start_pad_zeros, s[0].target))
        targ = pd.Series(tmp, name='ILI')

        # extract datelist from the first model, optionally converting to Saturdays
        tmp = s[0].datelist
        tmp = start_pad_dates + tmp
        time = pd.Series(tmp, name='Week')

        # concatenate into dataframe and output as csv
        df = pd.concat([time, targ] + preds, axis=1)
        df.to_csv(fname, index=False, date_format='%Y-%m-%d')

        if self.verbose is True:
            print '\n', datetime.datetime.now()
            print '----------------------END-----------------------'
