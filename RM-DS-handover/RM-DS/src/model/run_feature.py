import os
import argparse
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
import pandas as pd
import json
import numpy as np
from tqdm import *

import matplotlib.pylab as plt
from src.model.base_data import *
import shutil


EXPNAME = 'WINDOW7_HORIZON31_RESSTL'
FRAME = dict(window=7, horizon=31)
triples = pd.read_csv(os.path.join(PROCESSED, 'triples.csv'))
MCs = triples.MC.unique().tolist()
chunks = [MCs[x:x+6] for x in xrange(0, len(MCs), 6)]

def chunk2triple(which_chunk):
    chunk = chunks[which_chunk]
    return triples[triples.MC.isin(chunk)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', required=True, help='Baseline conf file')
    args = vars(parser.parse_args())
    chunk = int(args['chunk'])

    feature = AgregatedDataFeature(expname=EXPNAME,resume=True)

    triples = feature.identify_triples(chunk2triple(chunk))
    feature.triples = triples
    feature.gen_features()
