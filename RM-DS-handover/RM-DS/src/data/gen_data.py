'''
Script to gen the data in the PROCESSED directory

Take the different raw data files given by Nick,
Concatenate them and produced a file called
processed_data.csv in the processed folder of the data
folder.
'''

import os
import sys
import numpy as np
import pandas as pd
import argparse


# add the 'src' directory as one where we can import modules
sys.path.append(os.environ['ROOT_DIR'])
from setting import *


from src.data.helper import *


if __name__ == '__main__':

    name = 'data'

    filename = os.path.join(PROCESSED, 'processed_{}'.format(name))
    df = load_data()

    # Remove bad streams
    print('Collecting info on each stream')
    bigtable = gen_summary_table(df)

    # Select only good stream from main table
    print('Filtering all the bad streams from main table')
    df = filter_bad_stream(df, bigtable)

    print('-' * 50)
    print('Almost over, dump to disk')
    dump2csv(df, filename)
