'''

MIMO Transformer to put in the pipeline.
They should output arrays ;)

These transformer are able to transform time series
into feature suitable to use Machine Learning on top
of them.
'''

import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from dateutil.relativedelta import relativedelta
from dateutil.relativedelta import weekday
from sklearn.preprocessing import LabelEncoder


class IterateOverts(object):

    def iterate_over_ts(self, ts, window, horizon):
        nobs = len(ts) - horizon - window + 1
        for i in range(nobs):
            yield ts.iloc[i:i + window + horizon]


class MimoIDIdentifier(TransformerMixin, BaseEstimator, IterateOverts):

    '''
    Given a window and a horizon, return an array
    of the past window value for the given horizon

    '''

    def __init__(self, idx, window=7, horizon=7):
        ''' lags control the number of 
        temporal lag u want. Go until t-lag'''
        self.window = window
        self.horizon = horizon
        self.idx = idx

    def transform(self, X):
        nobs = len(X) - self.horizon - self.window + 1
        features = [self.idx] * nobs

        return np.array(features)[:, np.newaxis]

    def fit(self, X, y=None, **fit_params):
        return self


class MimoLagTransformer(TransformerMixin, BaseEstimator, IterateOverts):

    '''
    Given a window and a horizon, return an array
    of the past window value for the given horizon

    '''

    def __init__(self, window=7, horizon=7):
        ''' lags control the number of 
        temporal lag u want. Go until t-lag'''
        self.window = window
        self.horizon = horizon

    def transform(self, X):
        nobs = len(X) - self.horizon - self.window + 1
        features = []
        for df in self.iterate_over_ts(X, self.window, self.horizon):
            features.append(df.iloc[:self.window].values.flatten())

        return np.stack(features)

    def fit(self, X, y=None, **fit_params):
        return self


class MimoBHTransformer(TransformerMixin, BaseEstimator, IterateOverts):
    '''
    Look at the horizon and return a feature composed by two values.
    - The number of DOs in the horizon
    - A dumy arrays of every day in the horizon with 1 if it is BO and 0
    else
    '''

    def __init__(self, DO, window=7, horizon=7):
        ''' lags control the number of 
        temporal lag u want. Go until t-lag'''
        self.DO = DO
        self.window = window
        self.horizon = horizon

    def transform(self, X):
        nobs = len(X) - self.horizon - self.window + 1
        features = []
        for df in self.iterate_over_ts(X, self.window, self.horizon):
            h = df[self.window:]
            feats = map(int, h.index.isin(self.bos))
            features.append(np.array(feats).flatten())

        return np.stack(features)

    def fit(self, X, y=None, **fit_params):
        bh = pd.read_csv(os.path.join(PROCESSED, 'bankholliday.csv'),
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        self.bos = bh.groupby('DO').get_group(self.DO).date.tolist()
        return self


class MimoDaysTransformer(TransformerMixin, BaseEstimator, IterateOverts):
    '''
    Look at the horizon and return a feature composed by two values.
    - The day of week of the first day  and last day in the horizon
    - The day in month of the firs day and last day in the horizon
    - The day of year of the firs day and last day in the horizon
    - The month of the first and last day of the horizon
    - the weekof year of the first and last day of the horizon

    '''

    def __init__(self, window=7, horizon=7):
        ''' lags control the number of 
        temporal lag u want. Go until t-lag'''
        self.window = window
        self.horizon = horizon

    def get_first_last(self, h, freq='dayofweek'):
        # feats = [h.index[0], h.index[-1]]
        feats = map(lambda x: getattr(x, freq), h.index)
        return feats

    def transform(self, X):
        nobs = len(X) - self.horizon - self.window + 1
        features = []
        for df in self.iterate_over_ts(X, self.window, self.horizon):
            h = df[self.window:]
            feats = []
            # day of week
            feats.append(self.get_first_last(h, freq='dayofweek'))
            feats.append(self.get_first_last(h, freq='daysinmonth'))
            feats.append(self.get_first_last(h, freq='dayofyear'))
            feats.append(self.get_first_last(h, freq='month'))
            # feats.append(self.get_first_last(h, freq='weekofyear'))
            features.append(np.array(feats).flatten())

        return np.stack(features)

    def fit(self, X, y=None, **fit_params):
        return self


class MimoHollidayTransformer(TransformerMixin, BaseEstimator, IterateOverts):
    '''
    Look at the horizon and return a feature composed by two values.
    - The number of DOs in the horizon
    - A dumy arrays of every day in the horizon with 1 if it is BO and 0
    else
    '''

    def __init__(self, window=7, horizon=7):
        ''' lags control the number of 
        temporal lag u want. Go until t-lag'''
        self.window = window
        self.horizon = horizon

    def transform(self, X):
        nobs = len(X) - self.horizon - self.window + 1
        features = []
        for df in self.iterate_over_ts(X, self.window, self.horizon):
            h = df[self.window:]
            feats = map(lambda x: self.date2holliday[
                x] if x in self.date2holliday.keys() else 'Nothing', h.index)
            feats = self.label.transform(feats)
            features.append(np.array(feats).flatten())

        return np.stack(features)

    def fit(self, X, y=None, **fit_params):
        holliday = pd.read_csv(os.path.join(RAW, 'holliday.csv'),
                               parse_dates=['Date'],
                               date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
        holliday.columns = ['date', 'holliday', 'type', 'BH']
        holliday = holliday.set_index('date')
        self.date2holliday = holliday['holliday'].to_dict()
        self.label = LabelEncoder()
        self.label.fit(self.date2holliday.values() + ['Nothing'])
        return self


class MimoYTransformer(TransformerMixin, BaseEstimator, IterateOverts):
    '''
    Transformer to get the y values.
    '''

    def __init__(self, window=7, horizon=7):
        ''' lags control the number of 
        temporal lag u want. Go until t-lag'''
        self.window = window
        self.horizon = horizon

    def transform(self, X):
        nobs = len(X) - self.horizon - self.window + 1
        Xtmp = X.reset_index()
        features = []
        idxs = []
        for df in self.iterate_over_ts(X, self.window, self.horizon):
            h = df.iloc[self.window:]
            features.append(h.values.flatten())
            idxs.append(Xtmp[Xtmp['index'].isin(h.index)].index.tolist())
        return np.stack(features), np.stack(idxs)

    def fit(self, X, y=None, **fit_params):
        return self
