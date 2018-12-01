import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_decomposition import PLSRegression


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def lasso(X_train, y_train, X_pred):
    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)


def lasso_1se(X_train, y_train, X_pred):

    Ltest = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    Ltest.fit(X_train, y_train)

    # compute minimum cross-validation error + standard error as threshold
    alpha_index = np.where(Ltest.alphas_ == Ltest.alpha_)
    cv_means = np.mean(Ltest.mse_path_, axis=1)
    threshold = cv_means[alpha_index] + (np.std(Ltest.mse_path_[alpha_index]) / np.sqrt(10))

    # find highest alpha (sparsest model) with cross-validation error below the threshold
    alpha_new = max(Ltest.alphas_[np.where((Ltest.alphas_ >= Ltest.alpha_) & (cv_means < threshold))])

    # fit lasso with this alpha and predict training set
    Lpred = Lasso(alpha=alpha_new, max_iter=10000, tol=.0005, warm_start=True)

    return Lpred.fit(X_train, y_train).predict(X_pred)


def adaLasso(X_train, y_train, X_pred, gamma=1.1):

    alpha_grid = np.logspace(-1.5, 1, 20)
    estimator = RidgeCV(alphas=alpha_grid, cv=5)
    betas = estimator.fit(X_train, y_train).coef_

    weights = np.power(np.abs(betas), gamma)
    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train * weights, y_train).predict(X_pred * weights)


def adaLassoCV(X_train, y_train, X_pred):

    gamma = [0, 1.1, 2]
    alpha_grid = np.logspace(-1.5, 1, 20)
    estimator = RidgeCV(alphas=alpha_grid, cv=5)
    betas = estimator.fit(X_train, y_train).coef_

    mse_array = np.empty(len(gamma))
    for counter, g in enumerate(gamma):
        weights = np.power(np.abs(betas), g)
        lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
        training_fit = lr.fit(X_train * weights, y_train).predict(X_train * weights)
        mse_array[counter] = mse(training_fit, y_train)

    optimal_g = gamma[np.argmin(mse_array)]
    print np.argmin(mse_array)
    weights = 1 / np.power(np.abs(betas), optimal_g)
    prediction = lr.fit(X_train / weights, y_train).predict(X_pred / weights)

    return prediction


# ------------ WEIGHTINGS ------------- #
def lasso_obs(X_train, y_train, X_pred):
    X_train_wt = np.vstack((X_train, X_train[-4:, :]))
    y_train_wt = np.append(y_train, y_train[-4:])
    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train_wt, y_train_wt).predict(X_pred)

def lasso_obs_weighted(X_train, y_train, X_pred):
    X_train[:, [0, 1, 2, 52, 53, 54]] = X_train[:, [0, 1, 2, 52, 53, 54]] * 10
    X_pred[:, [0, 1, 2, 52, 53, 54]] = X_pred[:, [0, 1, 2, 52, 53, 54]] * 10

    X_train_wt = np.vstack((X_train, X_train[-4:, :]))
    y_train_wt = np.append(y_train, y_train[-4:])

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train_wt, y_train_wt).predict(X_pred)


def weighted_lasso(X_train, y_train, X_pred):
    # AR1,2,3 + top 3 exog.
    X_train[:, [0, 1, 2, 52, 53, 54]] = X_train[:, [0, 1, 2, 52, 53, 54]] * 10
    X_pred[:, [0, 1, 2, 52, 53, 54]] = X_pred[:, [0, 1, 2, 52, 53, 54]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred

def weighted_lasso2(X_train, y_train, X_pred):
    # AR1,2,3,52, first 5 exog.
    X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10
    X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred

def weighted_lasso2_obs(X_train, y_train, X_pred):
    # AR1,2,3,52, first 5 exog.
    X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_train[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10
    X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] = X_pred[:, [0, 1, 2, 51, 52, 53, 54, 55, 56]] * 10

    X_train_wt = np.vstack((X_train, X_train[-4:, :]))
    y_train_wt = np.append(y_train, y_train[-4:])

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train_wt, y_train_wt).predict(X_pred)
    return X_pred

def weighted_lasso3(X_train, y_train, X_pred):
    # AR1,2,3,6,12,52, first 3 exog.
    X_train[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] = X_train[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] * 10
    X_pred[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] = X_pred[:, [0, 1, 2, 5, 11, 51, 52, 53, 54]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred

def sort_weight_lasso(X_train, y_train, X_pred):
    # top 5 exog.
    X_train[:, [0, 1, 2, 3, 4]] = X_train[:, [0, 1, 2, 3, 4]] * 10
    X_pred[:, [0, 1, 2, 3, 4]] = X_pred[:, [0, 1, 2, 3, 4]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred


def ath_weight_lasso(X_train, y_train, X_pred):

    X_train[:, [52, 53, 54]] = X_train[:, [52, 53, 54]] * 10
    X_pred[:, [52, 53, 54]] = X_pred[:, [52, 53, 54]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred

def ath_ar_weight_lasso(X_train, y_train, X_pred):

    X_train[:, [0, 1, 2, 52, 53, 54]] = X_train[:, [0, 1, 2, 52, 53, 54]] * 10
    X_pred[:, [0, 1, 2, 52, 53, 54]] = X_pred[:, [0, 1, 2, 52, 53, 54]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred

def ath_deweight_lasso(X_train, y_train, X_pred):

    X_train[:, [52, 53, 54]] = X_train[:, [52, 53, 54]] * .1
    X_pred[:, [52, 53, 54]] = X_pred[:, [52, 53, 54]] * .1

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred


def wl1(X_train, y_train, X_pred):

    X_train[:, [0, 1, 2, 3, 12, 25, 51]] = X_train[:, [0, 1, 2, 3, 12, 25, 51]] * 10
    X_pred[:, [0, 1, 2, 3, 12, 25, 51]] = X_pred[:, [0, 1, 2, 3, 12, 25, 51]] * 10

    lr = LassoCV(cv=10, n_alphas=200, max_iter=10000, tol=.001, normalize=False)
    return lr.fit(X_train, y_train).predict(X_pred)
    return X_pred


def no_method(X_train, y_train, X_pred):
    ''' returns the input variable directly as the prediction.
    '''
    return X_pred



dispatcher = {
    'lasso': lasso,
    'lasso_1se': lasso_1se,
    'adalasso': adaLasso,
    'adalasso-cv': adaLassoCV,
    'None': no_method,
    'weighted-lasso': weighted_lasso,
    'weighted-lasso2': weighted_lasso2,
    'weighted-lasso3': weighted_lasso3,
    'lasso-obs': lasso_obs,
    'lasso-obs-weighted': lasso_obs_weighted,
    'weighted-lasso2-obs': weighted_lasso2_obs,
    'ath-weight': ath_weight_lasso,
    'ath-ar-weight': ath_ar_weight_lasso,
    'ath-deweight': ath_deweight_lasso,
    'wl1': wl1,
    'sort-weight': sort_weight_lasso,
    }
