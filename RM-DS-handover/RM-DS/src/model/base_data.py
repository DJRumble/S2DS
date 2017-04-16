'''
This file contains all the base class we use to work with the data.

RoyalMail is usefull to get some insight in the whole dataset
MC  : Looks at the scale of a mail center
Stream : Looks at the scale of a single triple (MC,DO stream).
AgregatedDataFeature : Usefulle to frame the problem as a ML one. Transforming time series in to feature matrix

'''


import os
import sys
sys.path.append(os.environ['ROOT_DIR'])
from setting import *
import pandas as pd
import json
import numpy as np
from tqdm import *
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from src.model.transformer import *
from src.model.mimo_transformer import *
import shutil
import pickle
import random
import json
import matplotlib.pylab as plt
import seaborn as sns


class RoyalMail(object):
    '''
    Helper class to look at the whole dataset.
    '''

    def __init__(self):
        self.df = self.load_df()
        self.MCs = self.df.MC.unique()
        self.DOs = self.df.DO.unique()
        self.nDOs = len(self.DOs)
        self.nMCs = len(self.MCs)
        self.streams = self.df.stream.unique()
        self.summary_table = self.load_summary_table()

    def dump_all_triples(self):
        '''
        Dump all possible combination (MC,DO,stream).
        More precisely, for each (MC,DO,stream), extract
        the sub dataframe from self.df and dump it to disk.
        Makes it easer to processe (MC,DO,stream) separately in the
        future
        '''
        gp = self.df.groupby(['MC', 'DO', 'stream'])
        N = gp.ngroups
        for (MC, DO, stream), df in tqdm(gp, total=N):
            name = '{}_{}_{}'.format('_'.join(MC.split(' ')),
                                     '_'.join(DO.split(' ')), stream)
            fname = os.path.join(PROCESSED, '{}.csv'.format(name))
            df.to_csv(fname, index=False)

    def load_df(self, chunksize=int(1e5)):
        '''
        Load the main cleaned data file located in the processed folder.
        '''
        df = pd.read_csv(os.path.join(PROCESSED, 'processed_data.csv'),
                         low_memory=False,
                         chunksize=chunksize,
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        n = 10389581  # size of df
        df = [tmp for tmp in tqdm(df, total=n / chunksize)]
        df = pd.concat(df)
        df.index = df.date
        return df

    def load_summary_table(self):
        '''
        Load the summary table showing statistics for each (MC,DO,stream)
        '''
        fname = os.path.join(PROCESSED, 'summary_table.csv')
        summary_table = pd.read_csv(fname)  # Return a plain dataframe
        # Create the nice Multiindex for clearer display
        summary_table.index = pd.MultiIndex.from_tuples(
            map(tuple, list(summary_table[['MC', 'DO', 'stream']].values)),
            names=['MC', 'DO', 'stream'])
        # Select only good columns
        summary_table = summary_table[
            [f for f in summary_table.columns if f not in ['MC', 'DO', 'stream']]]

        return summary_table


class MailCenter(object):
    '''
    Helper class to look at a specific mail center.

    '''

    def __init__(self, name, dump=False):
        '''
        Inputer:
        name : name of the mail centre in lower case
        dump : If true, the sub dataframe of processed_data.csv is
        going to be dump in the processed folder for faster reused
        in the future. Set it to true the first time you are looking
        at one mail center!

        '''
        self.mc = name
        self.name = '_'.join(name.split(' '))
        self.dump = dump
        self.df = self.load_df()
        if len(self.df.MC.unique()) != 1:
            raise ValueError('Error in the df')
        self.dos = self.df.DO.unique()
        self.nb_dos = len(self.dos)
        self.streams = self.df.stream.unique()
        self.mail_types = self.df.mail.unique()

    def gen_stream_stats(self, dos, target='stream', name='T0218'):
        '''
        Generate information for one stream and a list of dos.

        dos : List of dos
        target : 'stream' or 'mail'
        name : name of the specific target u want to study
        '''

        tmp = {do: self.df.groupby(['DO', target]).get_group(
            (do, name)).cnt.describe() for do in dos}
        tmp = pd.DataFrame(tmp).transpose()
        return tmp.sort_values('mean', ascending=False)

    def gen_DO_stats(self, do, target='stream'):
        '''
        Generate information for one do and different streams.

        do: name of the specific do u want to study
        target : specific target u want to study ('stream' or 'mail')

        return a dataframe witht the 

        '''
        tmp = {stream: self.df.groupby(['DO', target]).get_group(
            (do, stream)).cnt.describe() for stream in self.streams}
        tmp = pd.DataFrame(tmp).transpose()
        return tmp.sort_values('mean', ascending=False)

    def load_df(self):
        mc_fname = os.path.join(
            PROCESSED, '{}.csv'.format(self.name))
        if os.path.isfile(mc_fname) and not self.dump:
            df = pd.read_csv(mc_fname)
        else:
            print('*' * 50)
            print('Extracting the data in processed')
            df = pd.read_csv(os.path.join(PROCESSED, 'processed_data.csv'))
            df = df.groupby('MC').get_group(self.mc)
            df.to_csv(mc_fname, index=False)
        df.index = pd.to_datetime(df.date, format='%Y/%m/%d')
        df = df.sort_index()
        return df

    def get_DO(self, do, target='stream', name='T0076', drop_duplicates=True):
        '''
        Get a dataframe suming information for on specific target
        and on list of do

        do: list with do names
        '''

        if target not in ['stream', 'mail']:
            raise ValueError('target not appropriate')

        tmp = self.df.groupby(
            ['DO', target]).get_group((do, name))
        if drop_duplicates:
            tmp = tmp.groupby(lambda x: x.date()).aggregate(
                lambda x: sum(x))
        tmp.index = pd.DatetimeIndex(tmp.index)
        return tmp

    def compare_expected2given_count(self, do, target='stream', name='T0076'):
        '''
        For a specific tupple (do,target,name),
        print the expected number of record and the given number of record

        '''
        for year in ['2014', '2015', '2016']:
            tmp = self.get_DO(do, target=target,
                              name=name, drop_duplicates=False)[year]
            expe = pd.date_range(tmp.index.min(), tmp.index.max())
            print('-' * 50)
            print('The expeted length on {} with bank holliday,sunday.. is {}'.format(
                year, len(expe)))
            print('The length on {} is actually {}'.format(year, len(tmp)))

    def get_random_dos(self, size=10):
        indices = random.sample(range(self.nb_dos), size)
        dos_sample = [self.dos[i] for i in sorted(indices)]
        return dos_sample

    def plot_DO(self, dos, target='stream', name='T0076', ymax=30000):
        '''
        Plot target by do.

        target has to be either stream/mail
        '''

        if type(dos) == str:
            dos = [dos]

        fig = plt.figure(figsize=(20, 30))
        colors = sns.color_palette('deep', len(dos))
        for j, year in tqdm(enumerate(['2014', '2015', '2016']), total=3):
            ax = plt.subplot(3, 1, j + 1)
            for i, do in enumerate(dos):
                tmp = self.get_DO(do, target=target, name=name)
                try:
                    tmp[year].cnt.plot(ax=ax,
                                       label=do,
                                       lw=2,
                                       color=colors[i], marker='o')

                except:
                    pass
            ax.legend()

            ax.legend(fontsize=18)
            ax.tick_params(axis='both', which='major', labelsize=30)

    def plot_heatmap(self, stream='T0076', year='2015', resample=True):
        '''
        Plot a heat map of the count for a specific year and a specific stream

        Basically, it plots a matrix with all days (all week if resample is True)
        on the y axis and all do on the x axis.

        stream = Name of the stream
        year = Year u want to look at
        resample : Using Weekly resampling or not.

        '''

        df = self.get_DO(self.dos[0], name='T0076')[year]
        for i, do in tqdm(enumerate(self.dos[1:]), total=len(self.dos) - 1):
            try:
                df = df.join(self.get_DO(self.dos[i], name='T0076')[
                             year], rsuffix='_'.join(do.split(' ')))
            except:
                pass
        df = df[[f for f in df.columns if f[:3] == 'cnt']].fillna(0)
        if resample:
            df = df.resample('W').mean()

        # Noramlize for each do
        df = df / np.array(df.mean(axis=0))

        # Rename index
        df.index = map(lambda x: '{}_{}'.format(x.month, x.day), df.index)

        # Make figure
        fig = plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)
        sns.heatmap(df)

        return df


class Stream(object):

    def __init__(self, MC, DO, stream, dump=False):
        '''
        Look at a single triple (MC,DO,stream).
        Again use dump if you want to dump the df into disk.
        '''
        self.name = '{}_{}_{}'.format('_'.join(MC.split(' ')),
                                      '_'.join(DO.split(' ')), stream)
        self.fname = os.path.join(PROCESSED, '{}.csv'.format(self.name))
        self.DO = DO
        self.MC = MC
        self.stream = stream
        self.dump = dump
        self.rawdf = self.load_df()

    def check_on_duplicates(self, df):

        if len(df.index.get_duplicates()) == len(df) - len(df.drop_duplicates()):
            # print('Identify {} duplicates'.format(
            #     len(df.index.get_duplicates())))
            # print(
            #     'Safely remove them from df. They are just rows that have been inputed many times')
            df = df.drop_duplicates()
        else:
            raise ValueError('Identify some date duplicates that seems\
                             to contain different count. Take a closer look before any further processing')

        return df

    def load_df(self):
        if not os.path.isfile(self.fname) or self.dump:
            print('*' * 50)
            print('Extracting the data in processed')
            data = pd.read_csv(os.path.join(PROCESSED, 'processed_data.csv'))
            data = data.groupby(['MC', 'DO', 'stream']
                                ).get_group((self.MC, self.DO, self.stream))
            data.to_csv(self.fname, index=False)

        df = pd.read_csv(self.fname,
                         low_memory=False,
                         parse_dates=['date'],
                         date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
        df.index = df.date
        df.sort_index()
        df = self.check_on_duplicates(df)

        # # Merge on the baseline index
        date_range = pd.date_range(start='2014-04-01', end='2016-06-30')
        tmp = pd.DataFrame(index=date_range)
        df = tmp.merge(df, how='left', left_index=True, right_index=True)
        df.loc[:, 'day'] = map(lambda x: x.strftime('%A'), df.index)
        assert len(date_range) == len(df), 'Base date index is different'

        return df

    def inspect_df(self, df):
        '''
        print the expected number of record and the given number of record

        '''
        print('-' * 50)
        print('Expected/given df length')
        for year in ['2014', '2015', '2016']:
            tmp = df[year]
            expe = pd.date_range(tmp.index.min(), tmp.index.max())
            print('-' * 50)
            print('The expeted length on {} with bank holliday,sunday.. is {}'.format(
                year, len(expe)))
            print('The length on {} is actually {}'.format(year, len(tmp)))

    def train_test_split(self):
        '''
        Pipeline for comon processing of the data.
        Basically
        '''
        train = self.rawdf[START_TRAINING_DATE:END_TRAINING_DATE]
        test = self.rawdf[START_TESTING_DATE:END_TESTING_DATE]
        return train, test


class UtilsMixin(object):

    def init_folder(self, folder, override=False):
        if os.path.isdir(folder) and not override:
            raise ValueError('DIR Already exist')
        elif os.path.isdir(folder) and override:
            shutil.rmtree(folder)
            os.mkdir(folder)
        else:
            os.mkdir(folder)

    def dump2disk(self, dump, dump_name):
        with open(dump_name, 'wb') as f:
            pickle.dump(dump, f)

    def loadpickle(self, dump_name):
        with open(dump_name, 'rb') as f:
            dump = pickle.load(f)
        return dump

    def load_experiment(self, expname):
        fexpe = os.path.join(MODELS_DIR, expname, 'experiment.json')
        experiment = json.load(open(fexpe, 'r'))
        for key, val in experiment.items():
            setattr(self, key, val)

    def dump_experiment(self):

        exclude = ['MC_encoder', 'DO_encoder', 'stream_encoder']
        experiment = {key: val for key,
                      val in self.__dict__.iteritems() if key not in exclude}
        json.dump(experiment, open(os.path.join(
            self.folder, 'experiment.json'), 'w+'))

    def dump_pipelines(self, shared_pipeline, feat_pipeline):

        dump_name = os.path.join(self.folder, 'shared_pipeline.pickle')
        self.dump2disk(shared_pipeline, dump_name)
        dump_name = os.path.join(self.folder, 'feat_pipeline.pickle')
        self.dump2disk(feat_pipeline, dump_name)

    def load_pipelines(self):
        dump_name = os.path.join(self.folder, 'shared_pipeline.pickle')
        shared_pipeline = self.loadpickle(dump_name)
        dump_name = os.path.join(self.folder, 'feat_pipeline.pickle')
        feat_pipeline = self.loadpickle(dump_name)
        return shared_pipeline, feat_pipeline

    def identify_triples(self, data):
        df = data[['MC', 'DO', 'stream']].drop_duplicates()
        df = df.sort_values(['MC', 'stream'])
        df.index = range(len(df))
        triples = [tuple(x) for x in df.to_records(index=False)]
        if self.verbose:
            print('Identify {} triples to be fitted together'.format(len(df)))
        return triples

    def get_triple_name(self, MC, DO, stream):
        name = '{}_{}_{}'.format('_'.join(MC.split(' ')),
                                 '_'.join(DO.split(' ')), stream)
        return name


class AgregatedDataFeature(UtilsMixin):
    ''' 
    Class to handle one models for several MCs, DOs, streams specifically
    parameter:
        data : A dataframe with at least three columns - data/cnt/MC/DO/stream
    '''

    def __init__(self,
                 expname,
                 triples="",
                 dirname=PROCESSED,
                 frame=dict(window=14, horizon=31),
                 override=False,
                 resume=False,
                 verbose=False):

        self.resume = resume
        if not self.resume:
            self.dirname = dirname
            self.expname = expname
            self.verbose = verbose
            self.override = override
            self.frame = frame
            self.expname = expname
            self.triples = self.identify_triples(triples)
            self.init_encoder()
            self.folder = os.path.join(self.dirname, self.expname)
            self.init_folder(self.folder, self.override)
            self.init_pipelines()
            self.dump_experiment()
        else:
            self.load_experiment(os.path.join(dirname, expname))
            self.dirname = dirname
            self.expname = expname
            self.folder = os.path.join(self.dirname, self.expname)
            self.init_encoder()

    def init_encoder(self):
        df = pd.DataFrame(self.triples, columns=['MC', 'DO', 'stream'])
        self.MC_encoder = LabelEncoder().fit(df.MC)
        self.DO_encoder = LabelEncoder().fit(df.DO)
        self.stream_encoder = LabelEncoder().fit(df.stream)

    def init_pipelines(self):
        frame = self.frame
        shared_pipeline = Pipeline([
            ('RemoveOutlierTransformer', RemoveOutlierTransformer()),
            ('SundayRemover', SundayRemover()),
            ('BankHollidayTransformer', BankHollidayTransformer('')),
            ('SimpleFillInputer', SimpleFillInputer(0.0)),
            ('R', STLTransformer())
        ])
        feat_pipeline = FeatureUnion([
            ('Lags', MimoLagTransformer(**frame)),
            ('Days', MimoDaysTransformer(**frame)),
            ('BankHolliday', MimoBHTransformer(DO='', **frame)),
            # ('Holliday', MimoHollidayTransformer(**frame)),
            ('MCIdentifier', MimoIDIdentifier(idx='', **frame)),
            ('DOIdentifier', MimoIDIdentifier(idx='', **frame)),
            ('streamIdentifier', MimoIDIdentifier(idx='', **frame))
        ])
        self.dump_pipelines(shared_pipeline, feat_pipeline)

    def load_stacked_features(self, triples):
        triples = self.identify_triples(triples)
        Xtr = []
        Xte = []
        ytr = []
        yte = []
        error = []
        for MC, DO, stream in tqdm(triples, total=len(triples),
                                   disable=not self.verbose):
            try:
                train_feats, test_feats = self.load_feature(MC, DO, stream)
                Xtr.append(train_feats[1])
                ytr.append(train_feats[2])
                Xte.append(test_feats[1])
                yte.append(test_feats[2])
            except:
                error.append((MC, DO, stream))
        Xtr = np.vstack(Xtr)
        ytr = np.vstack(ytr)
        Xte = np.vstack(Xte)
        yte = np.vstack(yte)
        print('Found {} error'.format(len(error)))
        return Xtr, ytr, Xte, yte

    def gen_feature(self, MC, DO, stream):

        pipeline, features = self.load_pipelines()
        MC_id = self.MC_encoder.transform([MC])[0]
        DO_id = self.DO_encoder.transform([DO])[0]
        stream_id = self.stream_encoder.transform([stream])[0]

        streamObj = Stream(MC=MC, DO=DO, stream=stream)
        train, test = streamObj.train_test_split(pipeline=pipeline)
        pipeline.set_params(BankHollidayTransformer__DO=streamObj.DO)
        pipeline.set_params(RemoveOutlierTransformer__data=streamObj.rawdf)
        pipeline.set_params(RemoveOutlierTransformer__DO=streamObj.DO)
        pipeline.set_params(R__MC=streamObj.MC)
        pipeline.set_params(R__DO=streamObj.DO)
        pipeline.set_params(R__stream=streamObj.stream)
        pipeline.set_params(R__dump=True)
        pipeline.fit(train)
        train = pipeline.transform(train)
        test = pipeline.transform(test)

        features.set_params(DOIdentifier__idx=DO_id,
                            MCIdentifier__idx=MC_id,
                            streamIdentifier__idx=stream_id,
                            BankHolliday__DO=DO)

        X_train = features.fit_transform(train)
        X_test = features.transform(test)

        # labels
        labels = MimoYTransformer(**self.frame)
        y_train, idx_train = labels.fit_transform(train)
        y_test, idx_test = labels.transform(test)

        train_feat = (train, X_train, y_train, idx_train)
        test_feat = (test, X_test, y_test, idx_test)
        return train_feat, test_feat

    def gen_features(self):

        for MC, DO, stream in tqdm(self.triples, total=len(self.triples),
                                   disable=not self.verbose):
            try:
                train_feat, test_feat = self.gen_feature(MC, DO, stream)
                # dumping train
                dump_name = os.path.join(self.folder, '{}.train'.format(
                    self.get_triple_name(MC, DO, stream)))
                self.dump2disk(train_feat, dump_name)
                # dumping test
                dump_name = os.path.join(self.folder, '{}.test'.format(
                    self.get_triple_name(MC, DO, stream)))
                self.dump2disk(test_feat, dump_name)
            except:
                pass

    def load_feature(self, MC, DO, stream):
        dump_name = os.path.join(self.folder, '{}.train'.format(
            self.get_triple_name(MC, DO, stream)))
        train_feats = self.loadpickle(dump_name)
        dump_name = os.path.join(self.folder, '{}.test'.format(
            self.get_triple_name(MC, DO, stream)))
        test_feats = self.loadpickle(dump_name)
        return train_feats, test_feats
