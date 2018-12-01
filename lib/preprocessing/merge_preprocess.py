''' 
Load datasets from data directory and preprocess into a single csv for each state.

Input files are state_ili.csv containing all state ILI estimates,
GTdata_ST.csv with Google Trends information for each state ST,
and ATHdata.csv with athenahealth data for each state.

Processed results get saved in separate folders for each state.

Configuration: USE_DATA_FROM is the first week to start the data with.
               END_DATE is the final week to use in the dataset.


Fred Lu
6/6/17
'''

import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import ath_processor


def main():

    _ath_loc = config.ATH_LOC
    _targ_loc = config.STATE_ILI_LOC

    for ST in config.STATES:

        print 'Processing ' + ST

        _gt_loc = config.GT_DIR + '/GTdata_{0}.csv'.format(ST)
        _output_loc = config.STATES_DIR + '/{0}/{0}_merged_data.csv'.format(ST)

        try:
            # load files
            gt = pd.read_csv(_gt_loc, parse_dates=[0], index_col=[0])
            gt = gt.loc[:, (gt != 0).any(axis=0)]    # remove columns with all 0

            ath = ath_processor.ath_process(_ath_loc, ST, smooth=0)
            ath.set_index('week', inplace=True)

            targ = pd.read_csv(_targ_loc, parse_dates=[1], index_col=[1])
            targ = targ[[ST]]

            # merge dfs
            df = targ.merge(ath, how='outer', left_index=True, right_index=True)
            df2 = df.merge(gt, how='outer', left_index=True, right_index=True)

            # output
            df2 = df2.ix[pd.to_datetime(config.USE_DATA_FROM):pd.to_datetime(config.END_DATE)]
            df2.to_csv(_output_loc, date_format='%Y-%m-%d', index_label='Week')

            print 'Completed.'

        except Exception as e:
            print e


if __name__ == '__main__':
    main()
