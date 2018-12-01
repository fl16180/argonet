'''
Fred Lu 12/4/16
Accepts Athenahealth input file and processes it for a region
Output: csv
'''

from __future__ import division
import pandas as pd
import numpy as np


class Consts:
    INPUT = './data/ATHdata.csv'
    OUTPUT = './data/athena_MA.csv'
    SMOOTH = 0
    STATE = 'MA'


def ath_process(input_file, state, smooth=0):

    athena = pd.read_csv(input_file, parse_dates=[0])
    athena = athena.iloc[:, [0, 3, 4, 6, 7, 8, 9]]
    athena.columns = ['week', 'state', 'total', 'Flu', 'ILI', 'CDCILI', 'Viral']

    # select region here
    athena = athena[athena['state'] == state]

    if smooth == 1:
        # construct moving average of total visits
        _new = []
        for c in range(0, len(athena.total)):
            start = max(c - 104, 0)
            _new.append(np.mean(athena.total[start:c + 1]))
        athena['smooth total'] = _new

        # calculate athena rates using moving average of total visits
        rate = lambda x: x / athena['smooth total'] * 100
        athena['flu_var'] = rate(athena['Flu'])
        athena['ILI_var'] = rate(athena['ILI'])
        athena['viral_var'] = rate(athena['Viral'])

    elif smooth == 0:
        athena['athena_flu'] = athena['Flu'] / athena['total']
        athena['athena_ILI'] = athena['ILI'] / athena['total']
        athena['athena_viral'] = athena['Viral'] / athena['total']

    return athena[['week', 'athena_flu', 'athena_ILI', 'athena_viral']]


if __name__ == '__main__':

    athena = ath_process(Consts.INPUT, Consts.STATE, Consts.SMOOTH)

    athena.to_csv(Consts.OUTPUT, index=False)
