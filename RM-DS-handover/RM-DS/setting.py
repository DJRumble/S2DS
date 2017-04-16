import os
import sys

# ENV VARIABLES

ROOT_DIR = os.environ['ROOT_DIR']
# add the 'src' directory as one where we can import modules
SRC_DIR = os.path.join(os.environ['ROOT_DIR'], 'src')
DAT_DIR = os.path.join(os.environ['ROOT_DIR'], 'data')
MODELS_DIR = os.path.join(os.environ['ROOT_DIR'], 'models')

RAW = os.path.join(DAT_DIR, 'raw')
PROCESSED = os.path.join(DAT_DIR, 'processed')
INTERIM = os.path.join(DAT_DIR, 'interim')

FIGURES = os.path.join(ROOT_DIR, 'reports', 'figures')


START_DATE = '2014-04-06'
END_DATE = '2016-06-30'

START_TRAINING_DATE = START_DATE
END_TRAINING_DATE = '2016-04-11'
START_TESTING_DATE = '2016-04-12'
END_TESTING_DATE = END_DATE

# Model
RFCLUSTER = os.path.join(MODELS_DIR, 'RFCLUSTER')
RFCLUSTER_WINDOW7_FORECAST31 = os.path.join(
    MODELS_DIR, 'RFCLUSTER_WINDOW7_FORECAST31')
CLUSTER_WINDOW7_FORECAST31 = os.path.join(
    MODELS_DIR, 'CLUSTER_WINDOW7_FORECAST31')


TEST_MODEL = os.path.join(
    MODELS_DIR, 'TEST_MODEL')


STL_MODEL_0 = os.path.join(
    MODELS_DIR, 'STL_MODEL_0')

STL_MODEL = os.path.join(
    MODELS_DIR, 'STL_MODEL')

TBATS_MODEL = os.path.join(
    MODELS_DIR, 'TBATS_MODEL')

BENCHMARK_MODEL = os.path.join(MODELS_DIR, 'RM_BENCHMARK')

RF_TRIPLES = os.path.join(MODELS_DIR, 'RF_TRIPLES')

ANA_MODEL = os.path.join(MODELS_DIR, 'ANA_MODEL')

STLARIMA_MODEL = os.path.join(MODELS_DIR, 'STLARIMA')

EXTRATREE_BYMC_WINDOW7_HORIZON31_RESSTL = os.path.join(
    MODELS_DIR, 'EXTRATREE_BYMC_WINDOW7_HORIZON31_RESSTL')
