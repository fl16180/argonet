''' loading_functions.py
This file contains functions for loading variables from input files into 
the Geo_model object.

Dependencies: Requires location of CDC_REGIONS table from __init__.py, used in load_athena()

'''

from __future__ import division
import pandas as pd
import numpy as np
import sys

from __init__ import CDC_REGIONS


def date_selector(df, column, date_str):
    ''' accepts a time-series dataframe and returns the dataframe starting from 
    date given by date_str. column key indicates the column to parse.
    '''

    datelist = pd.to_datetime(df[column]).tolist()
    start = datelist.index(pd.to_datetime(date_str))

    return df.iloc[start:]


def load_singlefile(file_loc, date_col, target_col, **otherdata):
    ''' function to load all variables from a single data file.
        @params:
            file_loc: location of input file

            date_col: name or location of column containing dates. e.g. 'week' or [0]

            target_col: name or location of column containing target variable

            **otherdata: input other datasets in the same format as the above.
                
    Example: load_singlefile(file_loc='all.csv', date_col=[0], target_col=[1], 
                    gt=[3, 4, 5, 6], ath=['flu', 'ili', 'viral']) 
    '''

    all_file = pd.read_csv(file_loc)
    #all_file = date_selector(all_file, date_col, use_data_from)
    targ = all_file.iloc[:, target_col].values.astype(float)
    print 'Target loaded.'

    first_date = all_file.iloc[:, date_col].values[0]

    for name, value in otherdata.items():
        otherdata[name] = all_file.iloc[:, value].values.astype(float)
        print '{0} loaded'.format(name)

    return targ, otherdata, first_date


def load_target(file_loc, geo_level, geo_id, use_data_from):
    '''function to load target data from specified file_loc.
            For national and regional data, pre-written formats tailored
            to CDC ILINet files are used. Otherwise, a generic loader is used.
            output: n x 1 array
    '''

    if geo_level == 'National':
        targ_file = pd.read_csv(file_loc)
        targ_file = date_selector(targ_file, 'Date', use_data_from)
        targ = targ_file.iloc[:, 5].values.astype(float)

    elif geo_level == 'Regional':
        targ_file = pd.read_csv(file_loc)

        # Extract region number from REGION column
        temp = list(targ_file['REGION'])
        temp = [int(s[7:]) for s in temp]
        targ_file['REGION'] = temp

        # select specified region
        targ_file = targ_file[targ_file['REGION'] == geo_id]
        targ_file = date_selector(targ_file, 'Date', use_data_from)
        targ = targ_file.iloc[:, 5].values.astype(float)

    elif geo_level == 'State':
        targ_file = pd.read_csv(file_loc)

        state_data = targ_file[['Week', geo_id]]
        state_data = date_selector(state_data, 'Week', use_data_from)
        targ = state_data.iloc[:, 1].values.astype(float)

        # strip off nans only at end of array which is when a state's updates are behind
        while np.isnan(targ[-1]):
            targ = targ[:-1]

    elif geo_level == 'City':
        if geo_id == 'Boston':
            targ_file = pd.read_csv(file_loc)
            targ_file = date_selector(targ_file, 'Week beginning on', use_data_from)
            targ = targ_file.iloc[:, 1].values.astype(float)

    else:
        targ_file = pd.read_csv(file_loc)
        targ_file = date_selector(targ_file, 'Date', use_data_from)
        targ = targ_file.iloc[:, 1].values.astype(float)

    print "\tTarget file loaded."

    return targ


def load_gt(file_loc, use_data_from):
    '''function to load Google Trends API data from specified file_loc.
            output: n x p array, where p is number of Google variables
    '''
    gt_file = pd.read_csv(file_loc)
    gt_file = date_selector(gt_file, 'date', use_data_from)
    gt = gt_file.iloc[:, 1:].values

    print "\tGT file loaded."

    return gt


def load_fny(file_loc, use_data_from):
    '''function to load FNY API data from specified file_loc.
            output: n x 1 array
    '''
    fny_file = pd.read_csv(file_loc)
    fny_file = date_selector(fny_file, 'week', use_data_from)
    fny = fny_file.iloc[:, 1].values

    print "\tFNY file loaded."

    return fny


def load_athena(file_loc, geo_level, geo_id, use_data_from, smoothing=None):
    ''' input: uploaded Athenahealth data file.
            action: processes data and extracts relevant geographical location
            output: n x 3 numpy arrays
    '''
    athena = pd.read_csv(file_loc)

    if geo_level is 'Regional':

        # read in CDC region lookup table in order to merge states into regions
        CDC_regions = pd.read_csv(CDC_REGIONS, low_memory=False)

        # Merge HHS Regions for Athenahealth data and aggregate states
        athena = pd.merge(athena, CDC_regions, how='left', on='State', right_on=None,
                          left_index=False, right_index=False, sort=True, suffixes=('_x', '_y'), copy=True)
        athena = athena.groupby(['Region', 'Year', 'MMWR Week'], as_index=False).sum()

        # get data for specified region
        athena = athena[athena['Region'] == geo_id]

    elif geo_level is 'National':
        geo_id = 'ALL STATES'
        athena = athena[athena['State'] == geo_id]

    elif geo_level is 'State':
        athena = athena[athena['State'] == geo_id]

    else:
        # hard-coded exception for cities
        if geo_id == 'Boston':
            athena = athena[athena['State'] == 'MA']

    athena = athena[['Week Start Date', 'Visit Count', 'Flu Visit Count',
                     'ILI Visit Count', 'Unspecified Viral or ILI Visit Count']]
    athena = date_selector(athena, 'Week Start Date', use_data_from)

    if smoothing == 'moving-avg':
        # construct moving average of total case counts over 2 year period
        new = []
        for c in range(0, len(athena['Visit Count'])):
            start = max(c - 104, 0)
            new.append(np.mean(athena['Visit Count'][start:c + 1]))
        athena['smooth total'] = new

        # calculate athena rates using moving average of total visits
        rate = lambda x: x / athena['smooth total'] * 100
        athena['flu_smooth'] = rate(athena['Flu Visit Count'])
        athena['ILI_smooth'] = rate(athena['ILI Visit Count'])
        athena['viral_smooth'] = rate(athena['Unspecified Viral or ILI Visit Count'])
        ath = athena[['flu_smooth', 'ILI_smooth', 'viral_smooth']].values

    else:
        athena['flu_var'] = athena['Flu Visit Count'] / athena['Visit Count']
        athena['ILI_var'] = athena['ILI Visit Count'] / athena['Visit Count']
        athena['viral_var'] = athena['ILI Visit Count'] / athena['Visit Count']
        ath = athena[['flu_var', 'ILI_var', 'viral_var']].values

    print "\tAthena file loaded."

    return ath


def validate_vars(X, target):
    ''' checks input variable lengths against each other. Since date_selector has already
    been run on the vars, they will already be aligned. 
    Generally, ath and gt vars should be one longer than the target. gt is often an additional
    week longer because of the week-in-progress being reported, so this row is removed.

    This narrows down the various cases where one variable may not be updated using a simple set
    of rules that ignores various potential complications. Substitute validation functions can
    be written to handle these cases differently.
    '''

    try:
        # if length of gt is 2 longer than length of target, remove the last gt row
        if len(X['gt']) > len(target) + 1:
            tmp = X['gt']
            X['gt'] = tmp[:len(target) + 1]

        # check GT var length
        assert (len(X['gt']) == len(target) + 1)
    except AssertionError:
        print "Validation error: GT length - target length is: ", len(X['gt']) - len(target)
        sys.exit()
    except KeyError:
        pass

    try:
        # enforce ath length target + 1
        if len(X['ath']) > len(target) + 1:
            tmp = X['ath']
            X['ath'] = tmp[:len(target) + 1]

        assert (len(X['ath']) == len(target) + 1 or len(X['ath']) == len(target))
    except AssertionError:
        print "Validation error: ATH length - target length is: ", len(X['ath']) - len(target)
        sys.exit()
    except KeyError:
        pass

    try:
        # handles case where target and athena are same length, which could occur when the
        # target is already updated with T-1. Currently ignores the most recent target update.
        # also removes the most recent GT row based on ath since the previous check will not be
        # effective.
        if len(target) == len(X['ath']):
            if len(X['gt']) == len(X['ath']) + 1:
                tmp = X['gt']
                X['gt'] = tmp[:-1]

                target = target[:-1]
    except Exception as e:
        print e
        print "Error: target and ath lengths equal, gt length 1 longer"

    return X, target


def create_datelist(inputs, target, use_data_from, resolution):
    ''' creates an 'official' list of dates for the Geo_model object for reference
    with aligning all variables.
    '''

    # find the maximum length of all the input variables
    lengths = [len(inputs[xi]) for xi in inputs]
    lengths.append(len(target))

    # adds 3 to the maximum length to allow for up to T+2 horizon prediction
    listlen = max(lengths) + 3

    # constructs date list
    datelist = pd.date_range(start=use_data_from, periods=listlen, freq='7D').tolist()

    # TO-DO: Implement datelist for other resolutions besides week


    return datelist
