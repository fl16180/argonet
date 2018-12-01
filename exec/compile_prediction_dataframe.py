'''
Compiles each state's top predictions file into a single dataframe and exports to csv.
'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from lib import config


def compile(tag, outfile):

    VALID_STATES = ['AK','AL','AR','AZ','DE','GA','ID','KS','KY','LA','MA','MD','ME','MI',
                    'MN','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC',
                    'SD','TN','TX','UT','VA','VT','WA','WI','WV']

    print "Compiling tables into single file: "

    ens_df = []
    for i, state in enumerate(VALID_STATES):
        # print state

        state_dat = pd.read_csv(config.STATES_DIR + '/{0}/{1}.csv'.format(state, tag),
                                parse_dates=[0])
        state_dat['state'] = state

        ens_df.append(state_dat)

    ens_df = pd.concat([x for x in ens_df])

    ens_df.to_csv(config.STATES_DIR + '/_overview/{0}.csv'.format(outfile), index=False)


if __name__ == '__main__':

    TAG = 'top_argo_preds'
    OUTFILE = 'all_states_argo'

    compile(TAG, OUTFILE)
