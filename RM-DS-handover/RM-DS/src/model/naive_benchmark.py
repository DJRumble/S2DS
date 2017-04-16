import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
import pandas as pd
import json
import numpy as np
from tqdm import *

import matplotlib.pylab as plt
from src.model.base_data import Stream, RoyalMail
from src.model.transformer import *
from src.model.base_estimator import RMEstimatorMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import shutil


class SimpleNaiveModel(RMEstimatorMixin):

    '''
        Simple baseline for modelling.
        Use mean of the day in the previous years to
        make the forecast

        For instance, to predict 01/01/2016, we'll use
        the data available from the same day last year
    '''

    def __init__(self, DO):
        self.DO = DO
        self.day2int = dict(zip(['Monday', 'Tuesday', 'Wednesday',
                                 'Thursday', 'Friday', 'Saturday', 'Sunday'], range(7)))

    def get_mean_past_values(self, date):
        '''
        The tricks here is to get in the past year the value for the same day.
        They are not the same number of day in each year and then
        this can become tricky. The following works.

        '''
        day = date.strftime('%A')  # convert date to day
        wday = weekday(self.day2int[day])  # create a weekday instance
        # Find the value in the past
        past = date + relativedelta(years=-1, weekday=wday)
        # Get the value in the train.
        past_count = np.float(self.past.loc[past].cnt)
        return past_count

    def get_bos(self):
        ''' Return a list of dates which correspond to bo for this specific 
        do '''

        bh = pd.read_csv(os.path.join(PROCESSED, 'bankholliday.csv'),
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        return bh.groupby('DO').get_group(self.DO).date.tolist()

    def fit(self, X, y=None, **fit_params):
        '''
        Just memorized the training set
        during fit + load bank holliday

        parameter: X has to be a time series
        '''

        self.past = X
        self.bos = self.get_bos()
        return self.fit

    def predict(self, X, y=None, **predict_params):
        '''
        Use only past value to predict the future
        '''
        tmp = pd.DataFrame(X, columns=['cnt'])
        preds = map(lambda x: self.get_mean_past_values(x), tmp.index)
        preds = pd.DataFrame(preds, index=tmp.index)

        # Threeshold at zero any potentil bo
        preds[preds.index.isin(self.bos)] = 0
        return preds

    def forecast(self, X):
        df = self.predict(X)
        df.columns = ['yhat']
        return df


def init_folder(name, overide=False):
    folder = os.path.join(MODELS_DIR, name)
    if os.path.isdir(folder) and not overide:
        raise ValueError('DIR Already exist')
    elif os.path.isdir(folder) and overide:
        shutil.rmtree(folder)
        os.mkdir(folder)
    else:
        os.mkdir(folder)

if __name__ == '__main__':

    folder = os.path.join(MODELS_DIR, 'NAIVE_MODEL')
    init_folder('NAIVE_MODEL', overide=True)

    triples = pd.read_csv(os.path.join(PROCESSED, 'triples.csv'))
    N = len(triples)

    print('Forecast for {} triples'.format(N))
    error = []

    for i, MC, DO, stream in tqdm(triples.itertuples(), total=N):
        name = '{}_{}_{}'.format('_'.join(MC.split(' ')),
                                 '_'.join(DO.split(' ')), stream)
        fname = os.path.join(folder, name)

        try:
            stream = Stream(MC=MC,
                            DO=DO, stream=stream)
            pipeline = Pipeline([
                ('outliers', RemoveOutlierTransformer()),
                ('inputSunday', SundayTransformer()),
                ('inputBO', BankHollidayTransformer(stream.DO)),
                ('FillRemainingNan', SimpleFillInputer(0.0))
            ])
            train, test = stream.train_test_split(pipeline=pipeline)

            X_train = train
            y_train = X_train.cnt
            X_test = test
            y_test = X_test.cnt

            model = SimpleNaiveModel(DO=stream.DO)
            model.fit(X_train)

            forecast = model.forecast(X_test).join(y_test).reset_index()
            forecast.columns = ['date', 'yhat', 'ytrue']
            forecast['DO'] = DO
            forecast['MC'] = MC
            forecast['stream'] = stream

            forecast.to_csv(fname, index=False)
        except:
            error.append((MC, DO, stream))

    ferror = os.path.join(folder, 'error.csv')
    with open(ferror, 'w+') as f:
        for MC, DO, stream in error:
            f.write('{}_{}_{} \n'.format(MC, DO, stream))
