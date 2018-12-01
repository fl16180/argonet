from __future__ import division
import sys
import numpy as np
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import f_regression

from argo_methods import *


# scoring metric functions
def rmse(predictions, targets):
    # root mean square error
    return sqrt(((predictions - targets) ** 2).mean())


def corr(predictions, targets):
        # pearson r
    corr_c = pearsonr(predictions, targets)
    return corr_c[0]


def mae(predictions, targets):
    return np.absolute(predictions - targets).mean()


def mape(pred, targ):
    return np.mean(np.abs((targ - pred) / targ))


class ARGO(object):
    ''' wrapper for time-series machine learning predictions, known as ARGO
        (AutoRegressive model with General Online information)
    '''

    def __init__(self, X, y, transform):
        self.X = X
        self.y = y
        self.transform = transform
        self.predictions = []

    def make_predictions(self, method, horizon=0, training=104, in_sample=False, drop_nan=True,
                         normalize='stdev', transform_corr=False, bin_sparse=False, k_best=False,
                         drop_pred_zeros=False, drop_training_zeros=False):
        _counter = 0

        for week in xrange(training + horizon, len(self.X)):
            self._progress_indicator(_counter)
            _counter += 1

            # index vars for each week
            start = 0 if training == 'all' else max(0, week - (training + horizon))
            X_train = self.X[start:week - horizon]
            X_pred = self.X[week, None]
            y_train = self.y[start + horizon:week]

            # apply pre-processing
            if drop_nan is True:
                X_train, y_train, X_pred = self._drop_nan_features(X_train, y_train, X_pred)

            if drop_training_zeros is True:
                X_train, y_train = self.drop_training_target_zeros(X_train, y_train, self.transform)

            if bin_sparse is not False:
                X_train, X_pred = self._binarize_sparse_data(X_train, X_pred, threshold=bin_sparse, sum_counts=True)

            if drop_pred_zeros is True:
                X_train, X_pred = self._del_features_pred_zeros(X_train, X_pred)

            if transform_corr is True:
                X_train, X_pred = self._transform_by_corr(X_train, y_train, X_pred)

            if k_best is not False:
                X_train, X_pred = self._select_k_best_inorder(k_best, X_train, y_train, X_pred)

            if normalize is not False:
                if normalize == 'stdev':
                    X_train, X_pred = self._normalize_vars(X_train, X_pred)
                elif normalize == 'norm':
                    X_train, X_pred = self._center_normalize(X_train, X_pred)
                else:
                    print "Error: normalize must be one of 'stdev' or 'norm'."

            # run prediction function specified by 'method'
            self.predictions.append(dispatcher[method](X_train, y_train, X_pred))

        return np.asarray(self.predictions)

    def _normalize_vars(self, X_train, X_pred):
        ''' standardizes an array by subtracting mean and dividing by standard deviation.
        Handles columns of all zeros, returning all zeros for those columns rather than NaN.
        '''
        _ctrs = np.average(X_train, axis=0)
        _stds = np.std(X_train, axis=0)

        # handles all zeros
        _stds[_stds == 0.0] = 1.0

        X_train = (X_train - _ctrs) / _stds
        X_pred = (X_pred - _ctrs) / _stds

        return X_train, X_pred

    def _center_normalize(self, X_train, X_pred):
        ''' centers and normalizes an array with L2-norm.
        Also handles columns of all zeros, returning all zeros for those columns rather than NaN.
        '''
        _ctrs = np.average(X_train, axis=0)
        _norms = np.sqrt((X_train * X_train).sum(axis=0))

        # handles all zeros
        _norms[_norms == 0.0] = 1.0

        X_train = (X_train - _ctrs) / _norms
        X_pred = (X_pred - _ctrs) / _norms

        return X_train, X_pred

    def _drop_nan_features(self, X_train, y_train, X_pred):
        # remove any X features that contain NaNs
        mask1 = np.any(np.isnan(X_train), axis=0)
        mask2 = np.any(np.isnan(X_pred), axis=0)
        mask_f = mask1 + mask2

        X_train = X_train[:, ~mask_f]
        X_pred = X_pred[:, ~mask_f]

        # remove any y values that are NaN and the corresponding X row
        nan_y = np.isnan(y_train)
        y_train = y_train[~nan_y]
        X_train = X_train[~nan_y, :]

        return X_train, y_train, X_pred

    def drop_training_target_zeros(self, X_train, y, transform):
        # find rows for which target is 0 in target set and remove them

        if transform is True:
            target_zeros = y < -11.5
            y2 = y[~target_zeros]
            X_train = X_train[~target_zeros, :]
        else:
            target_zeros = y == 0
            y2 = y[~target_zeros]
            X_train = X_train[~target_zeros, :]

        return X_train, y2

    def _transform_by_corr(self, X_train, y_train, X_pred):

        X_train_new = np.empty(X_train.shape)
        X_pred_new = np.empty(X_pred.shape)

        for i in range(X_train_new.shape[1]):
            X_train_new[:, i], X_pred_new[:, i] = best_transform(X_train[:, i], y_train, X_pred[:, i])

        return X_train_new, X_pred_new

    def _del_features_pred_zeros(self, X_train, X_pred):
        mask = (X_pred == 0).reshape(-1)
        X_train = X_train[:, ~mask]
        X_pred = X_pred[:, ~mask]
        return X_train, X_pred

    def _select_k_best_inorder(self, k_best, X_train, y_train, X_pred, ar_cutoff=52):

        X_train_select = X_train[:, ar_cutoff:] if ar_cutoff is not None else X_train
        X_pred_select = X_pred[:, ar_cutoff:] if ar_cutoff is not None else X_pred

        f_scores, null = f_regression(X_train_select, y_train)
        k_best_f_scores = np.argsort(f_scores)[::-1][:k_best]

        X_train_best_sorted = X_train_select[:, k_best_f_scores]
        X_pred_best_sorted = X_pred_select[:, k_best_f_scores]

        if ar_cutoff is not None:
            X_train_best_sorted = np.hstack((X_train[:, :ar_cutoff], X_train_best_sorted))
            X_pred_best_sorted = np.hstack((X_pred[:, :ar_cutoff], X_pred_best_sorted))

        return X_train_best_sorted, X_pred_best_sorted

    def _binarize_sparse_data(self, X_train, X_pred, threshold=1, sum_counts=True):
        _zeros_per_column = np.sum(X_train == 0, axis=0)

        mask = (_zeros_per_column > threshold * X_train.shape[0])
        X_train[:, mask] = ~(X_train[:, mask] == 0)
        X_pred[:, mask] = ~(X_pred[:, mask] == 0)

        if sum_counts is True:
            nonzeros_per_week = -1 * np.sum(X_train == 0, axis=1).reshape(-1, 1)
            X_train = np.hstack((X_train, nonzeros_per_week))
            nonzeros_pred_week = -1 * np.sum(X_pred == 0, axis=1).reshape(-1, 1)
            X_pred = np.hstack((X_pred, nonzeros_pred_week))
        return X_train, X_pred

    def rm_sparse_data(arr, threshold=0.7):
        _zeros_per_column = np.sum(arr == 0, axis=0)

        mask = (_zeros_per_column > threshold * arr.shape[0])

        return arr[:, ~mask]


    def _progress_indicator(self, n):
        sys.stdout.write(str(n))
        sys.stdout.write('\r')
        sys.stdout.flush()


def best_transform(x, y, x_pred):
    a = lambda z: z ** .5
    b = lambda z: np.log(z + .5)
    c = lambda z: z ** 2
    d = lambda z: z

    fn_list = [a, b, c, d]
    transform_list = [a(x), b(x), c(x), d(x)]
    corr_list = [corr(xi, y) for xi in transform_list]

    best_transform = corr_list.index(max(corr_list))
    return transform_list[best_transform], fn_list[best_transform](x_pred)
