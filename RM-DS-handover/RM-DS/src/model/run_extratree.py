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
from src.model.base_estimator import Forecast
from sklearn.cross_validation import KFold
import argparse


import shutil

FOLDER = EXTRATREE_BYMC_WINDOW7_HORIZON31_RESSTL
EXPNAME = 'WINDOW7_HORIZON31_RESSTL'
FRAME = dict(window=7, horizon=31)

utils = UtilsMixin()



def write_summary(chunk,message, first=False):
    mode = 'a'
    if first:
        mode = 'w+'
    with open(os.path.join(FOLDER, 'summary_{}.txt'.format(chunk)), mode) as f:
        f.write(message + '\n')

def chunk2triple(which_chunk):
    chunk = chunks[which_chunk]
    return triples[triples.MC.isin(chunk)]            

triples = pd.read_csv(os.path.join(PROCESSED, 'triples.csv'))
MCs = triples.MC.unique().tolist()
chunks = [MCs[x:x+6] for x in xrange(0, len(MCs), 6)]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', required=True, help='Baseline conf file')
    parser.add_argument('--override', action='store_true')
    args = vars(parser.parse_args())
    chunk = int(args['chunk'])
    override = int(args['override'])
    if override:
        utils.init_folder(folder=os.path.join(FOLDER), override=True)

    feature = AgregatedDataFeature(expname=EXPNAME,resume=True)
    triples = chunk2triple(chunk)
    feature.triples = feature.identify_triples(triples)

    gp = triples.groupby('MC')
    N = gp.ngroups
    write_summary(chunk,'Forecast for {} groups'.format(N))
    
    for i, (name, df) in enumerate(gp):
        write_summary(chunk,'-' * 50)
        write_summary(chunk,'Remain stil {} to be fitted'.format(N - i))

        # specific_name = '_'.join(['_'.join(f.split(' ')) for f in name])
        specific_name = '_'.join(name.split(' '))
        
        write_summary(chunk,'Processing {}'.format(specific_name))
        model = ExtraTreeModel(folder=FOLDER,
                               frame=feature.frame,
                               specific_name=specific_name)
        
        Xtr, ytr, Xte, yte = feature.load_stacked_features(triples=df)
        
        write_summary(chunk,
            'Fitting the model on the all training set {}'.format(name))
        model.fit(Xtr, ytr, n_jobs=-1)
        # model.dump_model()

        write_summary(chunk,'Dumping the files for the forecast')
        forecast = Forecast(model=model,feature=feature)
        forecast.dump_forecasts(df)
