import pandas as pd

import matplotlib.pylab as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.model.base_estimator import BaseModel
from tqdm import *
import os

# from xgboost import XGBRegressor
# from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor


# class GradientBoostingTreeModel(BaseModel):

#     INITPARAMS = GradientBoostingRegressor().get_params()
#     name_model = 'gradientboostingtree'

#     def fit(self, X, y, **kwargs):
#         for key, value in kwargs.iteritems():
#             if key in self.INITPARAMS.keys():
#                 self.INITPARAMS[key] = value
#         model = MultiOutputRegressor(
#             GradientBoostingRegressor(**self.INITPARAMS))
#         model.fit(X, y)
#         self.model = model


class ExtraTreeModel(BaseModel):

    INITPARAMS = ExtraTreesRegressor().get_params()
    name_model = 'extratree'

    def fit(self, X, y, **kwargs):
        for key, value in kwargs.iteritems():
            if key in self.INITPARAMS.keys():
                self.INITPARAMS[key] = value
        model = ExtraTreesRegressor(**self.INITPARAMS)
        model.fit(X, y)
        self.model = model

    def predict_std(self, X):
        preds = []
        for estimator in self.model.estimators_:
            preds.append(estimator.predict(X))
        std = np.array(preds).std(axis=0)
        return std

    def forecast_std(self, feats):
        ts, X, y, idx_y = feats

        preds, error = self._average_forecast(
            self.predict_std(X), idx_y, len(ts))

        df = pd.DataFrame(np.stack((preds, error)).T,
                          columns=['yhat', 'error'],
                          index=ts.index)
        df.columns = ['stdhat', 'stderror']
        return df


# class XGBoostModel(BaseModel):

#     INITPARAMS = XGBRegressor().get_params()
#     name_model = 'xgboost'

#     def fit(self, X, y, **kwargs):
#         for key, value in kwargs.iteritems():
#             if key in self.INITPARAMS.keys():
#                 self.INITPARAMS[key] = value
#         model = MultiOutputRegressor(XGBRegressor(**self.INITPARAMS))
#         model.fit(X, y)
#         self.model = model


class RandomForestModel(BaseModel):

    INITPARAMS = RandomForestRegressor().get_params()
    name_model = 'randomforest'

    def fit(self, X, y, **kwargs):
        for key, value in kwargs.iteritems():
            if key in self.INITPARAMS.keys():
                self.INITPARAMS[key] = value
        model = RandomForestRegressor(**self.INITPARAMS)
        model.fit(X, y)
        self.model = model

    def predict_std(self, X):
        preds = []
        for estimator in self.model.estimators_:
            preds.append(estimator.predict(X))
        std = np.array(preds).std(axis=0)
        return std

    def forecast_std(self, MC, DO, stream):
        _, (ts, X, y, idx_y) = self.load_feature(MC, DO, stream)

        preds, error = self._average_forecast(
            self.predict_std(X), idx_y, len(ts))

        df = pd.DataFrame(np.stack((preds, error)).T,
                          columns=['yhat', 'error'],
                          index=ts.index)
        df.columns = ['stdhat', 'stderror']
        return df
