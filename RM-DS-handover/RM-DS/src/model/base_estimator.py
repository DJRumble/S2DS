import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import pandas as pd

import matplotlib.pylab as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import itertools
from tqdm import *
from random import shuffle

from sklearn.cross_validation import train_test_split as sk_train_test_split
from src.model.base_data import UtilsMixin


class Forecast(object):

    '''
    Class to ease the process of forecast given a model and some features

    - parameter
       - feature: an instance of AgregatedDataFeature to handle the feature u want
       to train with
       - model : an instance of BaseModel that u want to use to train on those feature
       to produce relevant forecast

    '''

    def __init__(self, feature, model):
        self.model = model
        self.feature = feature

    def residual(self, X, y, **residual_params):
        residual = self.model.predict(X).values.flatten() - X.values.flatten()
        residual = pd.DataFrame(residual,
                                index=X.index,
                                columns=['res'])
        return residual

    def plot_residual(self, X, y):
        fig = plt.figure(figsize=(20, 12))
        ax = plt.subplot(111)
        residual = self.residual(X, y)
        residual.res.plot(ax=ax)

    def _average_forecast(self, yhat, idxs, N):
        '''
        MIMO forecast comes in a sliding window format.
        This method aligne them correctly and them take the
        average as a final forecast.

        paramereters:
        - yhat: output of the model predict method 
        - idxs: corresponding indexes of each value in the original time
        series
        - N: Number of values in the original time serie.
        '''
        forecast = np.zeros((yhat.shape[0], N))

        mask = idxs.astype('int')
        for k in range(yhat.shape[0]):
            forecast[k, mask[k, :]] = yhat[k, :]
        forecast[np.where(forecast == 0)] = np.nan

        preds = np.nanmean(forecast.T, axis=1)
        error = np.nanstd(forecast.T, axis=1)

        return preds, error

    def forecast(self, MC, DO, stream):
        '''
        Given a MC, DO, stream triple, return a dataframe with three
        columns, namely,
        - the true value: ytrue
        - the predicted value: yhat
        - the error: error
        '''

        _, (ts, X, y, idx_y) = self.feature.load_feature(MC, DO, stream)

        preds, error = self._average_forecast(
            self.model.predict(X), idx_y, len(ts))

        df = pd.DataFrame(np.stack((preds, error)).T,
                          columns=['yhat', 'error'],
                          index=ts.index)
        ts = pd.DataFrame(ts, columns=['ytrue'])
        df = ts.join(df)
        df.columns = ['ytrue', 'yhat', 'error']
        return df

    def forecast_std(self, MC, DO, stream):
        '''
        Same method as forecast but used on the error to help
        have an estimate of the error on the forecast.

        Used mainly for Forest ensemble which return multiple tree, each one
        coming with its own prediction. An estimate of the error can then
        be obtained by looking at the standard deviation of the prediction
        which is what this method is looking at.

        Given a MC, DO, stream triple, return a dataframe with three
        columns, namely,
        - the true value: stdtrue
        - the predicted value: stdhat
        - the error: error
        '''

        _, (ts, X, y, idx_y) = self.feature.load_feature(MC, DO, stream)

        preds, error = self._average_forecast(
            self.model.predict_std(X), idx_y, len(ts))

        df = pd.DataFrame(np.stack((preds, error)).T,
                          columns=['yhat', 'error'],
                          index=ts.index)
        ts = pd.DataFrame(ts, columns=['ytrue'])
        df = ts.join(df)
        df.columns = ['ytrue', 'yhat', 'error']

        return df

    def stl_forecast(self, MC, DO, stream):
        '''
        Special mehod to handle forecast which are output with R.
        '''

        forecast = self.forecast(MC, DO, stream).reset_index()
        forecast.columns = ['date', 'stlres_ytrue',
                            'stlres_yhat', 'stlres_error']
        if hasattr(self.model, 'forecast_std'):
            std = self.forecast_std(MC, DO, stream)
            forecast['stlres_stdhat'] = std['yhat']
        forecast['DO'] = DO
        forecast['MC'] = MC
        forecast['stream'] = stream

        # stl
        stl = self.load_stl(MC, DO, stream).reset_index()
        final = stl.merge(forecast)
        final['stlhat'] = final['yhat']
        final['yhat'] = final['yhat'] + final['stlres_yhat']
        return final

    def dump_forecasts(self, df):
        '''
        Given a dataframe with at least three columns specifying the different
        triples (MC,DO,streal), dump the forecasts to disk
        '''
        error = []
        forecast_folder = os.path.join(self.model.folder, 'forecast')
        if not os.path.isdir(forecast_folder):
            os.mkdir(forecast_folder)

        triples = self.feature.identify_triples(df)
        for MC, DO, stream in tqdm(triples, total=len(triples)):
            try:
                fname = self.MCDOstream2name(MC, DO, stream)
                fname = os.path.join(forecast_folder, fname)
                forecast = self.stl_forecast(MC, DO, stream)
                forecast.to_csv('{}.csv'.format(fname), index=False)
            except:
                error.append((MC, stream, DO))
        ferror = os.path.join(forecast_folder, 'error.csv')
        with open(ferror, 'w+') as f:
            for MC, DO, stream in error:
                f.write('{}_{}_{} \n'.format(MC, DO, stream))

    def load_stl(self, MC, DO, stream):
        name = '{}_{}_{}'.format('_'.join(MC.split(' ')),
                                 '_'.join(DO.split(' ')), stream)
        stl = pd.read_csv(os.path.join(STL_MODEL, '{}.test'.format(name)),
                          parse_dates=['date'],
                          date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        return stl

    def MCDOstream2name(self, MC, DO, stream):
        name = '{}_{}_{}'.format('_'.join(MC.split(' ')),
                                 '_'.join(DO.split(' ')), stream)
        return name


class RMEstimatorMixin(BaseEstimator, RegressorMixin):

    def score(self, X, y):
        '''
        Function used by grid-search.
        Cross validation try to get thehighest value.
        That why there is - in front of the score
        '''

        return self.SMAPE(y, self.predict(X))

    def SMAPE(self, y, yhat):
        smape = np.abs(y - yhat) / ((y + yhat) / 2.0)
        smape = smape[smape < 1e10]
        return np.nanmean(smape) * 100

    def MAPE(self, y, yhat):
        return np.nanmean(np.abs(y - yhat) / (y + 1)) * 100

    def cross_val_score(self, X, y, iterator, **model_params):

        score = []
        for train_index, test_index in iterator:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.fit(X_train, y_train, **model_params)
            score.append(np.array(self.score(X_test, y_test)))
        mean = np.array(score).mean()
        std = np.array(score).std()
        return mean, std

    def dump_model(self):
        fmodel = os.path.join(
            self.folder, 'best_{}_{}.pkl'.format(self.name_model, self.specific_name))
        joblib.dump(self.model, fmodel)

    def load_model(self, name):
        fmodel = os.path.join(
            self.folder, 'best_{}_{}.pkl'.format(self.name_model, self.specific_name))
        self.model = joblib.load(fmodel)

    def grid_search_generator(self, grid_params):

        # adjust the format
        for key, p in grid_params.items():
            if type(p) == dict:
                grid_params[key] = [
                    f for f in range(p['min'], p['max'], 1)]

        # create the grid
        product = [x for x in itertools.product(*grid_params.values())]
        runs = [dict(zip(grid_params.keys(), p)) for p in product]
        shuffle(runs)
        for run in runs:
            yield run


class BaseModel(UtilsMixin, RMEstimatorMixin):

    '''
    Create the folder to contains model and everything needed.
    TO be used in combination with an AgregatedData instance.
    Make sure you have the same frame. Safest way is to
    instanciate the frame params with the aggregated
    instance

    '''

    def __init__(self, frame, folder, specific_name):
        self.frame = frame
        self.folder = folder
        self.specific_name = specific_name

    def predict(self, X):
        return self.model.predict(X)

    def gridsearch(self, X, y, grid, iterator=None):
        assert len(set(grid.keys()).intersection(
            self.INITPARAMS.keys())) == len(grid)
        model_params = self.INITPARAMS
        result = []
        N = reduce(lambda x, y: x * y, [len(f) for f in grid.values()])
        for params in tqdm(self.grid_search_generator(grid), total=N):
            model_params.update(params)
            if iterator is not None:
                score, std = self.cross_val_score(
                    X, y, iterator, **model_params)
            else:
                Xtr, Xval, ytr, yval = sk_train_test_split(
                    X, y, random_state=398)
                self.fit(Xtr, ytr)
                score = self.score(Xval, yval)
                std = 0
            res = model_params.copy()
            res['score'] = score
            res['std'] = std
            result.append(res)
        result = pd.DataFrame(result).sort_values('score')
        result.index = range(len(result))
        result.to_csv(os.path.join(
            self.folder, 'gridsearch_{}_{}.csv'.format(self.name_model, self.specific_name)), index=False)
        return result

    def get_best_grid_params(self, results):
        result = results.sort_values('score')
        result.index = range(len(result))
        best_params = result.iloc[0].to_dict()
        best_params = {key: value for key,
                       value in best_params.iteritems() if
                       key in self.INITPARAMS.keys()}
        return best_params
