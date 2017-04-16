'''

Transformer to put in the pipeline.
They should output arrays ;)

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

# R stuff
import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri as rpyn
pandas2ri.activate()
rforecast = importr('forecast')

# Pipeline stuff


class FeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, cols=[]):
        self.cols = cols

    def transform(self, X, **transform_params):
        tmp = X[self.cols]
        return tmp

    def fit(self, X, y=None, **fit_params):
        return self


class RemoveOutlierTransformer(BaseEstimator, TransformerMixin):
    '''
    Take a dataframe as input, return
    as df where the outlier have been replaced
    by the mean value over the same day two weeks before.
    '''

    def __init__(self, z=2, data=None, DO=""):
        '''
        z : number of sigma around the mean you want to use
        to detect outliers
        data : the original dataframe
        DO : the DO you are looking at.
        '''

        self.DO = DO
        self.z = z
        self.data = data

    def replace_outliers(self, X):
        # convert day to weekday index
        w2mean = X.groupby(X.index.week).median().cnt.to_dict()
        w2std = X.groupby(X.index.week).std().cnt.to_dict()

        new_count = []
        for i, row in X.iterrows():
            try:
                day = row.name.strftime('%A')
                wday = weekday(self.day2int[day])
                w = row.name.week
                mean = w2mean[w]
                std = w2std[w]
                if (day == 'Sunday' or row.name in self.bhs):
                    new_count.append(row.cnt)
                elif (abs((row.cnt - mean) / std) > self.z) or np.isnan(row.cnt):
                    days = [(row.name + relativedelta(weeks=i, weekday=wday))
                            for i in [-4, -3, -2, -1]]
                    new_count.append(self.data.loc[days].cnt.mean())
                else:
                    new_count.append(row.cnt)
            except:
                new_count.append(row.cnt)
        return new_count

    def transform(self, X, **transform_params):
        tmp = pd.DataFrame(X, columns=['cnt'])
        new_count = self.replace_outliers(tmp)
        tmp.loc[:, 'cnt'] = np.array(new_count)
        return tmp.cnt

    def fit(self, X, y=None, **fit_params):
        self.day2int = dict(zip(['Monday', 'Tuesday', 'Wednesday',
                                 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7)))
        bh = pd.read_csv(os.path.join(PROCESSED, 'bankholliday.csv'),
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        self.bhs = bh.groupby('DO').get_group(self.DO).date.tolist()
        return self


class SundayTransformer(BaseEstimator, TransformerMixin):
    '''
    Fill sunday with zeros.
    Specific to the cnt column!

    '''

    def transform(self, X, **transform_params):
        tmp = pd.DataFrame(X, columns=['cnt'])
        tmp['day'] = map(lambda x: x.strftime('%A'), tmp.index)
        tmp[tmp.day == 'Sunday'] = 0.0
        return tmp.cnt

    def fit(self, X, y=None, **fit_params):
        return self


class SundayRemover(BaseEstimator, TransformerMixin):
    '''
    Fill sunday with zeros.
    Specific to the cnt column!

    '''

    def transform(self, X, **transform_params):
        tmp = pd.DataFrame(X, columns=['cnt'])
        tmp['day'] = map(lambda x: x.strftime('%A'), tmp.index)
        tmp = tmp[tmp.day != 'Sunday']
        return tmp.cnt

    def fit(self, X, y=None, **fit_params):
        return self


class BankHollidayMeanInputer(BaseEstimator, TransformerMixin):
    '''
    Fill BH by mean
    '''

    def __init__(self, DO):
        self.DO = DO

    def replace_bkhd(self, X, row):
        date = row.name
        if date in self.bhs:
            D = date.strftime('%A')
            day = weekday(self.day2int[D])
            days = [(date + relativedelta(weeks=i, weekday=day))
                    for i in [-3, -2, -1]]
            try:
                val = X.loc[days].cnt.mean()
            except:
                val = X.cnt.mean()
        else:
            val = row.cnt
        return val

    def transform(self, X, **transform_params):
        tmp = pd.DataFrame(X, columns=['cnt'])
        tmp.cnt = [self.replace_bkhd(tmp, row) for i, row in tmp.iterrows()]
        return tmp.cnt

    def fit(self, X, y=None, **fit_params):
        self.day2int = dict(zip(['Monday', 'Tuesday', 'Wednesday',
                                 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7)))
        bh = pd.read_csv(os.path.join(PROCESSED, 'bankholliday.csv'),
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        self.bhs = bh.groupby('DO').get_group(self.DO).date.tolist()
        return self


class BankHollidayTransformer(BaseEstimator, TransformerMixin):
    '''
    Fill bank holliday with zeros
    '''

    def __init__(self, DO):
        self.DO = DO

    def transform(self, X, **transform_params):
        tmp = pd.DataFrame(X, columns=['cnt'])
        tmp[tmp.index.isin(self.bos)] = 0.0
        return tmp.cnt

    def fit(self, X, y=None, **fit_params):
        bh = pd.read_csv(os.path.join(PROCESSED, 'bankholliday.csv'),
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        self.bos = bh.groupby('DO').get_group(self.DO).date.tolist()
        return self


class SimpleFillInputer(TransformerMixin, BaseEstimator):
    '''
    Custom imputer function. Use fillna method from pandas instead
    of scikit learn.
    '''

    def __init__(self, fill=0):
        self.fill = fill

    def transform(self, X, **transform_params):
        return pd.DataFrame(X).fillna(value=self.fill)

    def fit(self, X, y=None, **fit_params):
        return self


class DenseTransformer(TransformerMixin, BaseEstimator):

    def transform(self, X):
        return X.todense()

    def fit(self, X, y=None):
        return self


# Forecasting with R


class TBATSReEstimator(TransformerMixin, BaseEstimator):
    ''' Implement STL residuals.

    Fit the model on the training and
    return the forecast for the test set.

    It returns the residual.

    THIS HAS TO BE USE WITH SUNDAY REMOVER ON TOP
    '''

    def __init__(self, triple_name="", dump=False):
        self.triple_name = triple_name
        self.dump = dump

    def transform(self, X):
        X = pd.DataFrame(X, columns=['cnt'])
        if X.index[0].year == 2014:
            tmp = X.cnt - self.tmp_train.yhat
            if self.dump:
                dump = X.join(self.tmp_train)
                self.dump_forecast(dump, split='train')
        else:
            tmp = X.cnt - self.tmp_test.yhat
            if self.dump:
                dump = X.join(self.tmp_test)
                self.dump_forecast(dump, split='test')
        return tmp

    def dump_forecast(self, tmp, split):
        if not os.path.isdir(TBATS_MODEL):
            os.mkdir(TBATS_MODEL)

        tmp = tmp.reset_index()
        if split == 'train':
            tmp.columns = ['date', 'ytrue', 'yhat']
        else:
            tmp.columns = ['date', 'ytrue', 'yhat',
                           'Lo80', 'Hi80', 'Lo95', 'Hi95']
        tmp.to_csv(os.path.join(
            TBATS_MODEL, '{}.{}'.format(self.triple_name, split)), index=False)

    def fit(self, X, y=None):
        tmp = pd.DataFrame(X, columns=['cnt'])
        train_dates = tmp.index
        test_dates = pd.DataFrame(index=pd.date_range(
            start=START_TESTING_DATE, end=END_TESTING_DATE))
        test_dates['day'] = map(lambda x: x.strftime('%A'), test_dates.index)
        test_dates = test_dates[test_dates.day != 'Sunday'].index
        h = len(test_dates)
        robj.globalenv['x'] = tmp
        robj.r('y = ts(x$cnt,frequency = 6)')
        robj.r('fit <- tbats(y,seasonal.period=c(6,313))')
        fitted = robj.r('fit$fitted.values')
        self.tmp_train = pd.DataFrame(
            fitted, columns=['yhat'], index=train_dates)
        forecast = robj.r(
            'fcast <- as.data.frame(forecast(fit, h = {}))'.format(h))
        forecast.columns = ['yhat', 'Lo80', 'Hi80', 'Lo95', 'Hi95']
        forecast.index = test_dates
        self.tmp_test = forecast
        return self


class STLEstimator(TransformerMixin, BaseEstimator):
    ''' Implement STL residuals.

    Fit the model on the training and
    return the forecast for the test set.

    It returns the residual.

    THIS HAS TO BE USE WITH SUNDAY REMOVER ON TOP
    '''

    def __init__(self, MC='', DO='', stream='', dump=False):
        self.dump = dump
        self.MC = MC
        self.DO = DO
        self.stream = stream

    def transform(self, X):
        X = pd.DataFrame(X, columns=['cnt'])
        if X.index[0].year == 2014:
            tmp = X.cnt - self.tmp_train.yhat
            if self.dump:
                dump = X.join(self.tmp_train)
                self.dump_forecast(dump, split='train')
        else:
            tmp = X.cnt - self.tmp_test.yhat
            if self.dump:
                dump = X.join(self.tmp_test)
                self.dump_forecast(dump, split='test')
        return tmp

    def dump_forecast(self, tmp, split):
        if not os.path.isdir(STL_MODEL):
            os.mkdir(STL_MODEL)

        tmp = tmp.reset_index()
        if split == 'train':
            tmp.columns = ['date', 'ytrue', 'yhat']
        else:
            tmp.columns = ['date', 'ytrue', 'yhat',
                           'Lo80', 'Hi80', 'Lo95', 'Hi95']

        tmp['MC'] = self.MC
        tmp['DO'] = self.DO
        tmp['stream'] = self.stream
        triple_name = '{}_{}_{}'.format('_'.join(self.MC.split(' ')),
                                        '_'.join(self.DO.split(' ')), self.stream)
        tmp.to_csv(os.path.join(
            STL_MODEL, '{}.{}'.format(triple_name, split)),
            index=False)

    def fit(self, X, y=None):
        tmp = pd.DataFrame(X, columns=['cnt'])
        train_dates = tmp.index
        test_dates = pd.DataFrame(index=pd.date_range(
            start=START_TESTING_DATE, end=END_TESTING_DATE))
        test_dates['day'] = map(lambda x: x.strftime('%A'), test_dates.index)
        test_dates = test_dates[test_dates.day != 'Sunday'].index
        h = len(test_dates)
        robj.globalenv['x'] = tmp
        robj.r('y = ts(x$cnt,frequency = 6)')
        robj.r('fit <- stl(y, s.window="periodic", robust=TRUE)')
        fitted = robj.r('fit$time.series')
        self.tmp_train = pd.DataFrame(np.sum(fitted[:, :2], axis=1), columns=[
                                      'yhat'], index=train_dates)
        forecast = robj.r(
            'fcast <- as.data.frame(forecast(fit, h = {}))'.format(h))
        forecast.columns = ['yhat', 'Lo80', 'Hi80', 'Lo95', 'Hi95']
        forecast.index = test_dates
        self.tmp_test = forecast
        return self
