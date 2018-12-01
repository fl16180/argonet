'''processing_functions.py

This file contains functions for processing already-loaded data.
These are called by a predictor object within Geo_model.

Functions include transforms and auto-regressive matrix creation.

Dependencies: None

'''

import numpy as np
from scipy.special import logit
from scipy.special import expit


def create_ar_stack(target, n):
    ''' creates autoregressive matrix from target array depending on input n.
        If n is integer, returns AR_n (matrix of most recent n values)
        If n is list, returns those specific AR terms (seasonal AR matrix)
    '''
    if isinstance(n, (int, long)):
        ar_mat = np.zeros((len(target) - n + 1, n))
        for t in range(n, len(target) + 1):
            ar_mat[t - n] = target[t - n:t]

    elif hasattr(n, '__len__') and not isinstance(n, str):
        ar_earliest = max(n)
        n = np.array(n)
        ar_mat = np.zeros((len(target) - ar_earliest + 1, len(n)))
        for t in range(ar_earliest, len(target) + 1):
            ar_mat[t - ar_earliest] = target[[t - n]]

    else:
        raise TypeError('Error: AR term must be int or list/array.')

    return ar_mat


######### TRANSFORMS #########

def logit_percent(data):
    n = len(data[data == 0])
    if n > 0:
        print '\t{0} values are 0, modifying for logit'.format(n)
    data[data == 0] = .001
    return logit(data / 100)


def gtlog(data):
    return np.log(data + .5)


def inverse_logit_percent(data):
    return 100 * expit(data)

# additional functions can be added here by re-implementing the transforms
# section as a dictionary following the format of argo_functions.py
