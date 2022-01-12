'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc
import glob
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
sample_dir_path = os.path.join(data_filedir, 'Samples')

def merge (sample_dir_path, unit_num):
    sample_array_lst = []
    label_array_lst = []

    for filepath in glob.glob(sample_dir_path + '/Unit' + unit_num + '*.npz'):
        print ("Loading %s ..." %filepath)
        # loaded = np.load(os.path.join(sample_dir_path, filepath))
        loaded = np.load(filepath)
        print(loaded['sample'].shape)
        print(loaded['sample'][0].dtype)
        print(loaded['label'].shape)
        print(loaded['label'][0].dtype)
        sample_array_lst.append(loaded['sample'])
        label_array_lst.append(loaded['label'])


    sample_array = np.dstack(sample_array_lst)

    label_array = np.concatenate(label_array_lst)

    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array


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


    # loaded = np.load(os.path.join(sample_dir_path,
    #                               'Unit%s_win%s_str%s.npz' % (str(int(unit_index)), sequence_length, stride)))
    # print(loaded['sample'].shape)
    # print(loaded['label'].shape)



    # sample_dir_files = os.path.join(sample_dir_path, '*.npz')
    dirFiles = os.listdir(sample_dir_path)  # list of directory files

    def SortingBigIntegers(arr, n):
        # Direct sorting using lamda operator
        # and length comparison
        arr.sort(key=lambda x: (len(x), x))

    sample_dir_files = dirFiles
    n = len(sample_dir_files)

    SortingBigIntegers(sample_dir_files, n)


    print (sample_dir_files)
    print (type(sample_dir_files))

    units_index_train = [2.0, 5.0, 10]
    # units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
    units_index_test = [11.0, 14.0, 15.0]

    sample_array_lst = []
    label_array_lst = []

    idx = units_index_train[0]
    sample_array, label_array = merge(sample_dir_path, str(int(idx)))

    sample_array = sample_array.transpose(2, 0, 1)
    print (sample_array.shape)




    # sample_array_lst = []
    # label_array_lst = []
    #
    # for filepath in glob.glob(sample_dir_path + '/Unit2*.npz'):
    #     print ("Loading %s ..." %filepath)
    #     # loaded = np.load(os.path.join(sample_dir_path, filepath))
    #     loaded = np.load(filepath)
    #     print(loaded['sample'].shape)
    #     print(loaded['label'].shape)
    #     sample_array_lst.append(loaded['sample'])
    #     label_array_lst.append(loaded['label'])
    #
    # sample_array = np.dstack(sample_array_lst)
    # label_array = np.concatenate(label_array_lst)

    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)



if __name__ == '__main__':
    main()
