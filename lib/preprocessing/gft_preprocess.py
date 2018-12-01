''' Preprocess Google Flu Trends data for each US state that has ILI data.
The relevant time period for GFT in this study is 2012-09-30 to 2015-08-09.

There are four options to scale GFT to state ILI over this time period:
    in-sample: GFT over this period is directly scaled to the corresponding ILI.

    fixed-two-year: The preceding two years of GFT is used to obtain a scaling constant.

    rolling-two-year: GFT for each week is re-scaled using the preceding two years.

    seasonal: GFT is re-scaled each year (defined as 20xx Week 40 to 20x(x+1) Week 39)
              using a scaling constant obtained from the previous year

Command-line input can be used to specify the option using '-i', '-f', '-r','-s', respectively.

Fred Lu
07/21/17
'''

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


# command-line options: 'python gft_preprocess.py -i'
options = {'-i': 'in-sample', '-f': 'fixed-two-year', '-r': 'rolling-two-year', '-s': 'seasonal'}

# load full GFT dataset, abbreviation lookup, and seasonal lookup
gft = pd.read_csv(config.DATA_DIR + '/GFT/GFT.csv', parse_dates=[0])
abbs = pd.read_csv(config.DATA_DIR + '/GFT/state_abbrev.csv')
seasons = pd.read_csv(config.DATA_DIR + '/ILI/CDC_season_lookup.csv', parse_dates=['Sunday'])

# options are: 'in_sample', 'fixed-two-year', 'rolling-two-year', 'seasonal'
try:
    SWITCH = options[sys.argv[1]]
except IndexError:
    SWITCH = 'fixed-two-year'

try:
    STATE = [sys.argv[2]]
except IndexError:
    print 'No state specified. Running all states...'
    STATE = abbs.ST

FIRST_PRED = '2012-09-30'
FIRST_PRED_MINUS_TWO_YEARS = '2010-10-03'


def main():

    # merge dataframes
    gft_seasons = gft.merge(seasons[['Sunday', 'Season']], how='inner',
                            left_on='Date', right_on='Sunday')
    gft_dates = gft_seasons.Date.tolist()

    for st in config.STATES:
        try:
            # select GFT time series for specific state
            state = abbs[abbs.ST == st].State.values
            gft_st = gft_seasons[state]

            # read ILI time series for specific state
            tmp = pd.read_csv(config.STATES_DIR + '/{0}/{0}_merged_data.csv'.format(st),
                              parse_dates=[0])
            ili = tmp[st]
            ili_dates = tmp.Week.tolist()

            # start all indexing from first training week (2 yrs before first prediction)
            gft_start = gft_dates.index(pd.to_datetime(FIRST_PRED_MINUS_TWO_YEARS))
            ili_start = ili_dates.index(pd.to_datetime(FIRST_PRED_MINUS_TWO_YEARS))

            gft_input = gft_st[gft_start:].values
            ili_input = ili[ili_start:].values
            datelist = gft_dates[gft_start:]

            # ili for printout for direct comparison
            ili_test = ili_input[104:len(gft_input)]

            # scale GFT with method based on switch
            if SWITCH == 'in-sample':
                gft_scaled = in_sample(gft_input, ili_input)

            elif SWITCH == 'fixed-two-year':
                gft_scaled = fixed_two_year(gft_input, ili_input)

            elif SWITCH == 'rolling-two-year':
                gft_scaled = rolling_two_year(gft_input, ili_input)

            elif SWITCH == 'rolling-one-year':
                gft_scaled = rolling_one_year(gft_input, ili_input)

            elif SWITCH == 'seasonal':
                # extact dates of each season start
                season_start_dates = []
                for year in [2011, 2012, 2013, 2014]:
                    val = seasons[(seasons.YEAR == year) & (seasons.WEEK == 40)].iloc[0]['Sunday']
                    season_start_dates.append(val)
                gft_scaled = seasonal(gft_input, ili_input, datelist, season_start_dates)

            else:
                print 'Error, switch does not match'

            # pad 0s to match global output format
            datelist_new = datelist[104:]
            start_pad_dates = pd.date_range(start=pd.datetime(2011, 1, 2),
                                            end=pd.to_datetime(FIRST_PRED) -
                                                pd.to_timedelta(1, unit='d'),
                                            freq='7D').tolist()
            start_pad_zeros = np.zeros(len(start_pad_dates))

            end_pad_dates = pd.date_range(start=pd.datetime(2015, 8, 16),
                                          end=pd.datetime(2017, 5, 15),
                                          freq='7D').tolist()
            end_pad_zeros = np.zeros(len(end_pad_dates))

            datelist_pred = start_pad_dates + datelist_new + end_pad_dates
            gft_pred_scaled = np.concatenate((start_pad_zeros, gft_scaled, end_pad_zeros))
            ili_test_scaled = np.concatenate((start_pad_zeros, ili_test, end_pad_zeros))

            week = pd.Series(datelist_pred, name='Week')
            signal = pd.Series(gft_pred_scaled, name='GFT')
            ili = pd.Series(ili_test_scaled, name='ILI')
            result = pd.concat([week, signal, ili], axis=1)
            result.to_csv(config.STATES_DIR + '/{0}/GFT_scaled.csv'.format(st), index=False)

        except Exception as e:
            print e


def in_sample(gft, ili):

    gft_all = gft[104:]
    ili_all = ili[104:104 + len(gft_all)]

    ols = LinearRegression()
    gft_pred_scaled = ols.fit(gft_all, ili_all).predict(gft_all)
    return gft_pred_scaled


def fixed_two_year(gft, ili):

    gft_train = gft[:104]
    gft_pred = gft[104:]
    ili_train = ili[:104]

    ols = LinearRegression()
    gft_pred_scaled = ols.fit(gft_train, ili_train).predict(gft_pred)
    return gft_pred_scaled

    # from sklearn.linear_model import Lasso
    # lin = Lasso(alpha=0.0001, positive=True)
    # gft_pred_scaled = lin.fit(gft_train, ili_train).predict(gft_pred)
    # return gft_pred_scaled


def rolling_two_year(gft, ili):

    gft_all = gft
    ili_all = ili[:len(gft_all)]

    ols = LinearRegression()
    gft_pred_scaled = []
    for i in range(104, len(gft_all)):
        gft_pred_scaled.append(ols.fit(gft_all[i - 104:i], ili_all[i - 104:i]).predict(gft[i, None]))
    return np.array(gft_pred_scaled).reshape(-1)


def rolling_one_year(gft, ili):

    gft_all = gft
    ili_all = ili[:len(gft_all)]

    ols = LinearRegression()
    gft_pred_scaled = []
    for i in range(52, len(gft_all)):
        gft_pred_scaled.append(ols.fit(gft_all[i - 52:i], ili_all[i - 52:i]).predict(gft[i, None]))
    return np.array(gft_pred_scaled).reshape(-1)[52:]


def seasonal(gft, ili, datelist, season_start_dates):

    ili = ili[:len(gft)]
    ols = LinearRegression()

    s2011 = datelist.index(season_start_dates[0])
    s2012 = datelist.index(season_start_dates[1])
    s2013 = datelist.index(season_start_dates[2])
    s2014 = datelist.index(season_start_dates[3])

    p2012 = ols.fit(gft[s2011:s2012], ili[s2011:s2012]).predict(gft[s2012:s2013])
    p2013 = ols.fit(gft[s2012:s2013], ili[s2012:s2013]).predict(gft[s2013:s2014])
    p2014 = ols.fit(gft[s2013:s2014], ili[s2013:s2014]).predict(gft[s2014:])

    gft_pred_scaled = np.concatenate((p2012, p2013, p2014))
    return gft_pred_scaled


if __name__ == '__main__':

    main()
