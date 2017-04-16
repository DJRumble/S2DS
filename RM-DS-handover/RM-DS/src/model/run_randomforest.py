import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
import pandas as pd
import json
import numpy as np
from tqdm import *

import matplotlib.pylab as plt
from src.model.base_data import Stream, RoyalMail, AgregatedDataFeature, UtilsMixin
from src.model.estimator import *
from sklearn.cross_validation import KFold


import shutil

FOLDER = RANDOMFOREST_BYSTREAM_WINDOW7_HORIZON31
FRAME = dict(window=7, horizon=31)
GRID = dict(n_estimators=[75, 100],
            max_features=[10, 20, 30])

utils = UtilsMixin()
utils.init_folder(folder=os.path.join(MODELS_DIR, EXPNAME), override=True)


def write_summary(message, first=False):
    mode = 'a'
    if first:
        mode = 'w+'
    with open(os.path.join(FOLDER, 'summary.txt'), mode) as f:
        f.write(message + '\n')


def dump_forecasts(feature, model, df):
    error = []

    forecast_folder = os.path.join(model.folder, 'forecast')
    if not os.path.isdir(forecast_folder):
        os.mkdir(forecast_folder)
    triples = feature.identify_triples(df)
    for MC, DO, stream in tqdm(triples, total=len(triples)):
        try:
            fname = model.MCDOstream2name(MC, DO, stream)
            fname = os.path.join(forecast_folder, fname)
            _, test_features = feature.load_feature(MC, DO, stream)
            forecast = model.forecast(test_features).reset_index()
            forecast.columns = ['date', 'ytrue', 'yhat', 'error']
            if hasattr(model, 'forecast_std'):
                std = model.forecast_std(test_features)
                forecast['stdhat'] = std['stdhat']
            forecast['DO'] = DO
            forecast['MC'] = MC
            forecast['stream'] = stream
            forecast.to_csv('{}.csv'.format(fname), index=False)
        except:
            error.append((MC, stream, DO))
    ferror = os.path.join(forecast_folder, 'error.csv')
    with open(ferror, 'w+') as f:
        for MC, DO, stream in error:
            f.write('{}_{}_{} \n'.format(MC, DO, stream))


if __name__ == '__main__':

    triples = pd.read_csv(os.path.join(PROCESSED, 'triples.csv'))
    gp = triples.groupby(['MC', 'stream'])
    N = gp.ngroups
    write_summary('Forecast for {} groups'.format(N))

    # Init instances
    feature = AgregatedDataFeature(expname='WINDOW7_HORIZON31',
                                   resume=True)
    for i, (name, df) in enumerate(gp):
        write_summary('-' * 50)
        write_summary('Remain stil {} to be fitted'.format(N - i))
        specific_name = '_'.join(['_'.join(f.split(' ')) for f in name])
        write_summary('Processing {}'.format(specific_name))
        model = RandomForestModel(folder=FOLDER,
                                  frame=feature.frame,
                                  specific_name=specific_name)
        Xtr, ytr, Xte, yte = feature.load_stacked_features(triples=df)
        write_summary('Gridsearch for {}'.format(name))
        res = model.gridsearch(Xtr, ytr, grid=GRID)
        best_params = model.get_best_grid_params(res)
        write_summary(
            'Refitting the model on the all training set {}'.format(name))
        model.fit(Xtr, ytr, **best_params)
        model.dump_model()

        write_summary('Dumping the files for the forecast')
        dump_forecasts(feature, model, df)
