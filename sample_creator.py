'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc
import argparse
import os
import json
import logging
import sys
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import random
import importlib
from scipy.stats import randint, expon, uniform
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
import scipy.stats as stats
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
# import keras
import tensorflow as tf
print(tf.__version__)
# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.data_preparation import df_all_creator, df_train_creator, df_test_creator, Input_Gen

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
seed = 0
random.seed(0)
np.random.seed(seed)
# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')




def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=10, help='Input sources', required=True)
    parser.add_argument('-s', type=int, default=10, help='sequence length')
    parser.add_argument('--index', type=int, default='non', help='data representation:non, sfa or pca')


    args = parser.parse_args()

    sequence_length = args.w
    stride = args.s
    unit_index = args.index



    # Load data
    '''
    W: operative conditions (Scenario descriptors)
    X_s: measured signals
    X_v: virtual sensors
    T(theta): engine health parameters
    Y: RUL [in cycles]
    A: auxiliary data
    '''

    df_all = df_all_creator(data_filepath)

    '''
    Split dataframe into Train and Test
    Training units: 2, 5, 10, 16, 18, 20
    Test units: 11, 14, 15

    '''
    # units = list(np.unique(df_A['unit']))
    units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
    units_index_test = [11.0, 14.0, 15.0]

    print("units_index_train", units_index_train)
    print("units_index_test", units_index_test)

    # if any(int(idx) == unit_index for idx in units_index_train):
    #     df_train = df_train_creator(df_all, units_index_train)
    #     print(df_train)
    #     print(df_train.columns)
    #     print("num of inputs: ", len(df_train.columns) )
    #     df_test = pd.DataFrame()
    #
    # else :
    #     df_test = df_test_creator(df_all, units_index_test)
    #     print(df_test)
    #     print(df_test.columns)
    #     print("num of inputs: ", len(df_test.columns))
    #     df_train = pd.DataFrame()


    df_train = df_train_creator(df_all, units_index_train)
    print(df_train)
    print(df_train.columns)
    print("num of inputs: ", len(df_train.columns) )
    df_test = df_test_creator(df_all, units_index_test)
    print(df_test)
    print(df_test.columns)
    print("num of inputs: ", len(df_test.columns))

    del df_all
    gc.collect()
    df_all = pd.DataFrame()
    sample_dir_path = os.path.join(data_filedir, 'Samples')

    cols_normalize = df_train.columns.difference(['RUL', 'unit'])
    sequence_cols = df_train.columns.difference(['RUL', 'unit'])

    data_class = Input_Gen (df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                            unit_index, stride =stride)
    data_class.seq_gen()



if __name__ == '__main__':
    main()
