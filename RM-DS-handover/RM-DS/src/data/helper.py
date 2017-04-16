'''
Helper to handle the data
'''

import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
import pandas as pd
import json
import numpy as np
from tqdm import *

SIZE_FILE = {'s2ds1.txt': 4896774,
             's2ds2.txt': 2463409,
             's2ds3.txt': 4151883,
             's2ds4.txt': 2034817,
             's2ds1_update.txt': 2114958}


def dump2csv(df, file_name, bychunk=False):
    if bychunk:
        df.to_csv('{}.csv'.format(file_name),
                  index=False)
    else:
        df.to_csv('{}.csv'.format(file_name),
                  index=False,
                  chunksize=int(1e5))
    print('Successfully dump the file to {}'.format(file_name))


def load_one_file(name, chunksize=1e5):
    '''
    Load only one txt file in a dataframe
    '''
    print('Loading file {} '.format(name))
    fname = os.path.join(RAW, name)
    df = pd.read_csv(os.path.join(RAW, name),
                     low_memory=False,
                     chunksize=int(1e5),
                     sep='\t',
                     header=0,
                     names=["stream", "mail", "MC", "DO", "cnt", "date"],
                     parse_dates=['date'],
                     date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    df = pd.concat([tmp for tmp in tqdm(
        df, total=SIZE_FILE[name] / chunksize)])
    df.MC = [x.lower() for x in df.MC]
    df.DO = [x.lower() for x in df.DO]
    df.mail = [x.lower() for x in df.mail]
    df.index = df.date
    df = df.sort_index()
    df['day'] = [x.strftime('%A') for x in df.index]
    return df


def load_data():
    df1 = load_one_file('s2ds1.txt')
    # df1new = load_one_file('s2ds1_update.txt')
    df2 = load_one_file('s2ds2.txt')
    df3 = load_one_file('s2ds3.txt')
    df4 = load_one_file('s2ds4.txt')

    df = [df1[START_DATE:], df2, df3, df4]

    print('-' * 50)
    print('Common processing step. Sorry, take a while')
    # Then we wan to take the intersection of all these files
    print('Indentify intersection')

    # MC
    all_MC = reduce(lambda x, y: set(y.MC.unique()).union(
        set(x)), df[1:], df[0].MC.unique())
    common_MC = reduce(lambda x, y: set(y.MC.unique()).intersection(
        set(x)), df[1:], df[0].MC.unique())
    bad_MC = all_MC.difference(common_MC)

    # DO
    all_DO = reduce(lambda x, y: set(y.DO.unique()).union(
        set(x)), df[1:], df[0].DO.unique())
    common_DO = reduce(lambda x, y: set(y.DO.unique()).intersection(
        set(x)), df[1:], df[0].DO.unique())
    bad_DO = all_DO.difference(common_DO)

    # Now we identified the good ones, take only the record in these MC
    # and these DOs
    print('Pick the good row in df ')
    df = [f[~f.MC.isin(bad_MC)] for f in tqdm(df, total=len(df))]
    df = [f[~f.DO.isin(bad_DO)] for f in tqdm(df, total=len(df))]

    print('Concat the remaining rows')
    df = pd.concat(df)
    df = df.sort_index()

    return df


def gen_summary_table(df):

    fname = os.path.join(PROCESSED, 'rm_bigtable.csv')
    gp = df.groupby(['MC', 'DO', 'stream'])
    cols = ['mean', 'max', 'min', 'std', 'count']
    cnts = gp.cnt.aggregate(['mean', 'max', 'min', 'std', 'count'])
    cnts.columns = ['cnt_{}'.format(f) for f in cnts]
    dates = gp.date.aggregate(['min', 'max'])
    dates.columns = ['date_{}'.format(f) for f in dates]
    bigtable = cnts.join(dates)
    return bigtable


def filter_bad_stream(df, bigtable):
    date_max = df.date.max()
    date_min = df.date.min()
    cond = (
        (bigtable.date_max == date_max) &
        (bigtable.date_min == date_min) &
        (bigtable.cnt_count < 915) &
        (bigtable.cnt_count > 600)
    )
    # Dump good table to disk
    bigtable[cond].to_csv(os.path.join(PROCESSED, 'summary_table.csv'))
    good_streams = bigtable[cond].index.tolist()
    gp = df.groupby(['MC', 'DO', 'stream'])
    df = pd.concat([gp.get_group(group)
                    for group in tqdm(good_streams)])
    return df


def dump_bank_holliday():

    df = pd.read_csv(os.path.join(RAW, 'DOsandBOv2.csv'),
                     low_memory=False,
                     header=0,
                     names=["DO", "region", "date", "day", "bengland",
                            "bscotland", 'bnorthern_ireland'],
                     parse_dates=['date'],
                     date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y'))
    df.DO = [f.lower() for f in df.DO]
    df.day = [f.lower() for f in df.day]
    df.region = [str(f).lower() for f in df.region]
    df.region = map(lambda x: 'northern_ireland' if x ==
                    'northern ireland' else x, df.region)
    df.benland = [bool(f) for f in df.bengland]
    df.bscotland = [bool(f) for f in df.bscotland]
    df.bnorthern_ireland = [bool(f) for f in df.bnorthern_ireland]

    tmp = []
    for region in tqdm(['england', 'scotland', 'northern_ireland']):
        d = df[(df['b{}'.format(region)]) & (df.region == region)]
        d = d.drop_duplicates()
        tmp.append(d[['DO', 'date']])

    tmp = pd.concat(tmp)
    tmp['bankholliday'] = True
    print('Dump bank hollday')
    tmp.to_csv(os.path.join(PROCESSED, 'bankholliday.csv'), index=False)
