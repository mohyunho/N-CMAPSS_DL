'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt


from utils.elm_network import network_fit

from utils.hpelm import ELM, HPELM
from utils.elm_task import SimpleNeuroEvolutionTask
from utils.ea_multi import GeneticAlgorithm


# random seed predictable
jobs = 1
seed = 0
random.seed(seed)
np.random.seed(seed)


current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'elm_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')

pic_dir = os.path.join(current_dir, 'Figures')
ea_log_path = os.path.join(current_dir, 'EA_log')



'''
load array from npz files
'''
def load_part_array (sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge (sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array (sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array


def load_array (sample_dir_path, unit_num, win_len, stride):
    filename =  'Unit%s_win%s_str%s.npz' %(str(int(unit_num)), win_len, stride)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']


def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    print("ind_list befor: ", ind_list[:10])
    print("ind_list befor: ", ind_list[-10:])
    ind_list = shuffle(ind_list)
    print("ind_list after: ", ind_list[:10])
    print("ind_list after: ", ind_list[-10:])
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label



def score_calculator(y_predicted, y_actual):
    # Score metric
    h_array = y_predicted - y_actual
    s_array = np.zeros(len(h_array))
    for j, h_j in enumerate(h_array):
        if h_j < 0:
            s_array[j] = math.exp(-(h_j / 13)) - 1

        else:
            s_array[j] = math.exp(h_j / 10) - 1
    score = np.sum(s_array)
    return score


def release_list(a):
   del a[:]
   del a



units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
units_index_test = [11.0, 14.0, 15.0]



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='RPs creator')
    parser.add_argument('-w', type=int, default=1, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-constant', type=float, default=1e-4, help='constant for #neurons penalty')
    parser.add_argument('-bs', type=int, default=1000, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=20, help='patience')
    parser.add_argument('-vs', type=float, default=0.1, help='validation split')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')
    parser.add_argument('--device', type=str, default="GPU", help='Use "basic" if GPU with cuda is not available')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    lr = args.lr
    cs = args.constant
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub

    device = args.device


    train_units_samples_lst =[]
    train_units_labels_lst = []

    for index in units_index_train:
        print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride)
        #sample_array, label_array = shuffle_array(sample_array, label_array)
        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        sample_array = sample_array[::sub]
        label_array = label_array[::sub]

        sample_array = sample_array.astype(np.float32)
        label_array = label_array.astype(np.float32)

        print("sub sample_array.shape", sample_array.shape)
        print("sub label_array.shape", label_array.shape)
        train_units_samples_lst.append(sample_array)
        train_units_labels_lst.append(label_array)

    sample_array = np.concatenate(train_units_samples_lst)
    label_array = np.concatenate(train_units_labels_lst)
    print ("samples are aggregated")

    release_list(train_units_samples_lst)
    release_list(train_units_labels_lst)
    train_units_samples_lst =[]
    train_units_labels_lst = []
    print("Memory released")

    sample_array, label_array = shuffle_array(sample_array, label_array)
    print("samples are shuffled")
    print("sample_array.shape", sample_array.shape)
    print("label_array.shape", label_array.shape)

    sample_array = sample_array.reshape(sample_array.shape[0], sample_array.shape[2])
    print("sample_array_reshape.shape", sample_array.shape)
    print("label_array_reshape.shape", label_array.shape)
    feat_len = sample_array.shape[1]
    num_samples = sample_array.shape[0]
    print ("feat_len", feat_len)

    train_sample_array = sample_array[:int(num_samples*(1-vs))]
    train_label_array = label_array[:int(num_samples*(1-vs))]
    val_sample_array = sample_array[int(num_samples*(1-vs))+1:]
    val_label_array = label_array[int(num_samples*(1-vs))+1:]

    print ("train_sample_array.shape", train_sample_array.shape)
    print ("train_label_array.shape", train_label_array.shape)
    print ("val_sample_array.shape", val_sample_array.shape)
    print ("val_label_array.shape", val_label_array.shape)


    corr_log_path = os.path.join(current_dir, 'corr_log.csv')
    corr_log_col = ['params_1', 'params_2', 'params_3', 'params_4', 'train_params']
    corr_log_df = pd.DataFrame(columns=corr_log_col, index=None)
    corr_log_df.to_csv(corr_log_path, index=False)

    num_params = np.arange(5, 205, 5, dtype=int)
    lin_vec = np.ones(int(len(num_params)), dtype=int) * 2
    lr_lst = np.ones(int(len(num_params)), dtype=int)*5

    print(lr_lst)
    corr_log_df['params_1'] = lr_lst
    corr_log_df['params_2'] = num_params
    corr_log_df['params_3'] = num_params
    corr_log_df['params_4'] = lin_vec
    corr_log_df['train_params'] = num_params * 20

    print(corr_log_df)
    train_time_lst = []
    test_time_lst = []

    hof = []
    for index, p_ind in corr_log_df.iterrows():
        hof.append(p_ind[0:4].values)

    """ Creates a new instance of the training-validation task and computes the fitness of the current individual """

    for i in range(len(hof)):

        # if i == 0:
        #     hof_ref = "min"
        # else:
        #     hof_ref = "med"

        l2_parms_lst = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        l2_parm = l2_parms_lst[hof[i][0] - 1]
        type_neuron_lst = ["tanh", "sigm", "lin"]

        lin_check = hof[i][3]
        num_neuron_lst = []
        for n in range(2):
            num_neuron_lst.append(hof[i][n + 1] * 10)
        if lin_check == 1:
            num_neuron_lst.append(20)
        else:
            num_neuron_lst.append(0)

        print("HoF l2_params: ", l2_parm)
        print("HoF lin_check: ", lin_check)
        print("HoF num_neuron_lst: ", num_neuron_lst)
        print ("dtype(num_neuron_lst[0])", type(num_neuron_lst[0]))
        print("HoF type_neuron_lst: ", type_neuron_lst)

        num_neuron_lst = [int(s) for s in num_neuron_lst]

        feat_len = train_sample_array.shape[1]
        best_elm_class = network_fit(feat_len,
                              l2_parm, lin_check,
                              num_neuron_lst, type_neuron_lst, model_temp_path, device, bs)
        best_elm_net = best_elm_class.trained_model()

        # Train the best network
        sample_array = np.concatenate((train_sample_array, val_sample_array))
        label_array = np.concatenate((train_label_array, val_label_array))

        start = time.time()

        print ("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        best_elm_net.train(sample_array, label_array, "R")
        print("individual trained...evaluation in progress...")
        neurons_lst, norm_check = best_elm_net.summary()
        print("summary: ", neurons_lst, norm_check)

        end = time.time()

        train_time = end - start
        train_time_lst.append(train_time)

        output_lst = []
        truth_lst = []

        # Test
        start = time.time()
        for index in units_index_test:
            print ("test idx: ", index)
            sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride)
            # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
            print("sample_array.shape", sample_array.shape)
            print("label_array.shape", label_array.shape)
            sample_array = sample_array[::sub]
            label_array = label_array[::sub]
            print("sub sample_array.shape", sample_array.shape)
            print("sub label_array.shape", label_array.shape)
            sample_array = sample_array.reshape(sample_array.shape[0], sample_array.shape[2])
            print("sample_array_reshape.shape", sample_array.shape)
            print("label_array_reshape.shape", label_array.shape)

            sample_array = sample_array.astype(np.float32)
            label_array = label_array.astype(np.float32)

            # estimator = load_model(model_temp_path)

            y_pred_test = best_elm_net.predict(sample_array)
            output_lst.append(y_pred_test)
            truth_lst.append(label_array)

        print(output_lst[0].shape)
        print(truth_lst[0].shape)

        print(np.concatenate(output_lst).shape)
        print(np.concatenate(truth_lst).shape)

        output_array = np.concatenate(output_lst)[:, 0]
        trytg_array = np.concatenate(truth_lst)
        print(output_array.shape)
        print(trytg_array.shape)

        output_array = output_array.flatten()
        print(output_array.shape)
        print(trytg_array.shape)
        score = score_calculator(output_array, trytg_array)
        print("score: ", score)

        rms = sqrt(mean_squared_error(output_array, trytg_array))
        print(rms)
        rms = round(rms, 2)
        score = round(score, 2)

        end = time.time()
        test_time = end - start
        test_time_lst.append(test_time)

        print("train_time: ", train_time)
        print("test_time: ", test_time)
        print("HOF phenotype: ", [l2_parms_lst[hof[i][0] - 1], hof[i][1] * 10, hof[i][2] * 10, hof[i][3]])
        print(" test RMSE: ", rms)

    corr_log_df['train_time'] = train_time_lst
    corr_log_df['test_time'] = test_time_lst
    corr_log_df.to_csv(corr_log_path, index=False)

    fig_verify = plt.figure(figsize=(8, 5))
    plt.plot(corr_log_df['train_params'] , corr_log_df['train_time'] )
    plt.xticks(corr_log_df['train_params'] )
    plt.ylabel("Train time", fontsize=13)
    plt.xlabel("Trainable parameters", fontsize=13)
    fig_verify.savefig(os.path.join(pic_dir, 'corr_lin_plot_%s_%s.png'))


if __name__ == '__main__':
    main()
