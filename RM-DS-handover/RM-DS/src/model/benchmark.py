import os
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

import seaborn as sns

START_BENCHMARK = '2016-05-09'
END_BENCHMARK = '2016-05-31'


class Results(object):

    def __init__(self, name,
                 label,
                 include_all=False,
                 start_date=START_BENCHMARK,
                 end_date=END_BENCHMARK):
        self.include_all = include_all
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.label = label
        self.fname = os.path.join(MODELS_DIR, '{}.csv'.format(name))
        if self.include_all:
            self.fname = os.path.join(
                MODELS_DIR, '{}_alldates.csv'.format(name))
        if os.path.isfile(self.fname):
            self.df = pd.read_csv(self.fname)
        else:
            print('The file does not exist')
        self.rectify_yhat()

    def SMAPE(self, y, yhat):
        return np.nanmean(np.abs(y - yhat) / ((y + yhat) / 2.0)) * 100

    def rectify_yhat(self):
        self.df = self.df[self.df.ytrue != 0]
        self.df['yhat'] = map(lambda x: 0 if x < 0 else x, self.df.yhat)

    def dump_file(self, folder, endswith):
        res = []
        listfiles = [f for f in os.listdir(
            folder) if f.endswith(endswith)]
        for f in tqdm(listfiles, total=len(listfiles)):
            try:
                res.append(pd.read_csv(os.path.join(folder, f),
                                       parse_dates=['date'],
                                       date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d')))
            except:
                pass

        forecast = pd.concat(res)
        forecast = forecast.set_index('date')
        if not self.include_all:
            forecast = forecast[(forecast.index >= self.start_date)
                                & (forecast.index <= self.end_date)]
        forecast = forecast.reset_index()
        forecast.to_csv(self.fname, index=False)

    def SMAPE_distribution(self):
        d = self.df.groupby(['MC', 'DO', 'stream']).apply(
            lambda x: self.SMAPE(x.ytrue, x.yhat)).reset_index()
        d.columns = ['MC', 'DO', 'stream', 'SMAPE']
        return d


class ModelComparison(object):

    def __init__(self, models):
        self.models = models
        self.stream2mail = json.load(
            open(os.path.join(PROCESSED, 'stream2mail.json')))

    def SMAPE(self, y, yhat):
        return np.nanmean(np.abs(y - yhat) / ((y + yhat) / 2.0)) * 100

    def merge(self):
        dfs = []
        for model in self.models:
            name = '_'.join('_'.join(model.label.split(' ')).split('+'))
            on = ['MC', 'DO', 'stream', 'date', 'ytrue']
            cols = [f for f in model.df.columns if f not in on]
            cols_new = ['{}_{}'.format(f, name) for f in cols]
            left = model.df[on]
            right = model.df[cols]
            right.columns = cols_new
            dfs.append(left.join(right))
        df = reduce(lambda left, right:
                    left.merge(right, on=on), dfs)
        df['yhat_ensemble'] = (df.yhat_STL +
                               df.yhat_STL_ARIMA + df.yhat_STL_Rforest) / 3
        df['mail'] = map(lambda x: self.stream2mail[x], df.stream)
        return df

    def get_SMAPE_distrib(self):
        dists = []
        for i, model in tqdm(enumerate(self.models), total=len(self.models)):
            dist = model.SMAPE_distribution()
            dist['model'] = model.label
            dists.append(dist)

        dists = reduce(lambda a, b: a.append(b), dists[1:], dists[0])
        return dists

    def distribution_SMAPE(self, name_fig):
        fig = plt.figure(figsize=(20, 10))
        ax = plt.subplot(111)
        c = sns.color_palette('deep', len(self.models))
        for i, model in tqdm(enumerate(self.models), total=len(self.models)):
            dist = model.SMAPE_distribution()
            ax = sns.kdeplot(dist['SMAPE'], shade=True,
                             color=c[i], label=model.label)
            ax.axvline(float(dist.SMAPE.mean()), ymin=0,
                       ymax=1, ls='--', lw=4, color=c[i])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 0.07)
        ax.set_xlabel('Relative error % (SMAPE)', size=32)
        ax.set_ylabel('Frequency', size=32)
        ax.legend(fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=30)
        name = os.path.join(FIGURES, "{}.png".format(name_fig))
        fig.savefig(name)

    def distribution_SMAPE_violin(self, name_fig, palette, width=20):
        fig = plt.figure(figsize=(width, 10))
        ax = plt.subplot(111)
        SMAPES = []
        names = []
        for i, model in tqdm(enumerate(self.models), total=len(self.models)):
            dist = model.SMAPE_distribution()
            names.append([model.label] * len(dist.SMAPE))
            SMAPES.append(dist.SMAPE.tolist())
        SMAPES = reduce(lambda a, b: a + b, SMAPES)
        names = reduce(lambda a, b: a + b, names)
        dist = pd.DataFrame(
            np.array([SMAPES, names]).T, columns=['SMAPE', 'model'])
        dist.SMAPE = map(float, dist.SMAPE)
        ax = sns.violinplot(x="model", y="SMAPE", data=dist,
                            inner="quartile", palette=palette)
        ax.set_ylim(0, 80)
        ax.set_xlabel('Models', size=32)
        ax.set_ylabel('Relative error %', size=32)
        ax.legend(fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=30)
        name = os.path.join(FIGURES, '{}.png'.format(name_fig))
        fig.savefig(name)

    def distribution_SMAPE_violin_BYmail(self, name_fig):
        fig = plt.figure(figsize=(20, 10))
        ax = plt.subplot(111)
        c = sns.color_palette('deep', 5)
        SMAPES = []
        names = []
        df = self.merge()
        col2name = {'yhat_ensemble': 'Best model',
                    'yhat_RM_Benchmark': 'Benchmark'}
        for i, model in tqdm(enumerate(['yhat_RM_Benchmark', 'yhat_ensemble']), total=2):
            dist = df.groupby(('MC', 'DO', 'stream', 'mail')).apply(
                lambda x: self.SMAPE(x.ytrue, x[model])).reset_index()
            dist.columns = ['MC', 'DO', 'stream', 'mail', 'SMAPE']
            dist['model'] = col2name[model]
            SMAPES.append(dist)
        SMAPES = SMAPES[0].append(SMAPES[1])
        SMAPES.SMAPE = map(float, SMAPES.SMAPE)
        ax = sns.violinplot(x="mail", y="SMAPE", hue="model", data=SMAPES, split=True,
                            inner="quart", palette={"Best model": [0.73, 0.13, 0.13], "Benchmark": 'grey'})
        sns.despine(left=True)
        ax.set_ylim(0, 80)
        ax.set_xlabel('Mail formats', size=32)
        ax.set_ylabel('Relative error %', size=32)
        ax.legend(fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=30)
        name = os.path.join(FIGURES, '{}.png'.format(name_fig))
        fig.savefig(name)


class BestModel(object):

    def __init__(self, models):
        self.models = models
        self.stream2mail = json.load(
            open(os.path.join(PROCESSED, 'stream2mail.json')))
        self.df = self.merge()

    def merge(self):
        dfs = []
        for model in self.models:
            name = '_'.join('_'.join(model.label.split(' ')).split('+'))
            on = ['MC', 'DO', 'stream', 'date']
            cols = [f for f in model.df.columns if f not in on]
            cols_new = ['{}_{}'.format(f, name) for f in cols]
            left = model.df[on]
            right = model.df[cols]
            right.columns = cols_new
            dfs.append(left.join(right))
        df = reduce(lambda left, right:
                    left.merge(right, on=on), dfs)
        df['yhat_ensemble'] = (df.yhat_STL + df.yhat_STL_Rforest) / 2.0
        df['mail'] = map(lambda x: self.stream2mail[x], df.stream)
        return df

    def get_triple(self, MC, DO, stream):

        df = self.df.groupby(['MC', 'DO', 'stream']
                             ).get_group((MC, DO, stream))
        df.date = map(lambda x: pd.to_datetime(x), df.date)
        df = df.set_index('date')
        return df
