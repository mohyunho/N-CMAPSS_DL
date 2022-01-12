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


# directory_path = current_dir + '/EA_log'
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

    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="GPU", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="moo", help='Use "soo" for single objective and "moo" for multiobjective')

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
    obj = args.obj

    pop = args.pop
    gen = args.gen



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

    col_a = 'fitness_1'
    col_b = 'fitness_2'

    mute_log_file_path = os.path.join(ea_log_path, 'mute_log_%s_%s.csv' % (pop, gen))
    # ea_log_path + 'mute_log_%s_%s.csv' % (pop_size, n_generations)
    mute_log_df = pd.read_csv(mute_log_file_path)
    print(mute_log_df)
    prft_log_file_path = os.path.join(ea_log_path, 'prft_out_%s_%s.csv' % (pop, gen))
    # ea_log_path + 'mute_log_%s_%s.csv' % (pop_size, n_generations)
    prft_log_df = pd.read_csv(prft_log_file_path, header=0, names=["p1", 'p2', 'p3', 'p4'])
    # prft_log_df = pd.read_csv(prft_log_file_path)
    print(prft_log_df)

    fit1_lst = []
    fit2_lst = []

    for index, p_ind in prft_log_df.iterrows():
        # print ("index", index)
        # print ("p_ind", p_ind)
        # print ("p_ind['p1']", p_ind['p1'])
        log_prft_ind = mute_log_df.loc[(mute_log_df['params_1'] == p_ind['p1']) &
                                       (mute_log_df['params_2'] == p_ind['p2']) &
                                       (mute_log_df['params_3'] == p_ind['p3']) &
                                       (mute_log_df['params_4'] == p_ind['p4'])]

        print("log_prft_ind", log_prft_ind)
        fit1_lst.append(log_prft_ind[col_a].values[0])
        fit2_lst.append(log_prft_ind[col_b].values[0])


    prft_log_df[col_a] = fit1_lst
    prft_log_df[col_b] = fit2_lst

    print("prft_log_df", prft_log_df)

    min_fit1 = np.min(prft_log_df[col_a])
    print("min_fit1", min_fit1)
    min_prft_ind = prft_log_df.loc[(prft_log_df[col_a] == min_fit1)]
    print("min_prft_ind", min_prft_ind)

    med_fit2 = np.median(prft_log_df[col_b])
    print("med_fit2", med_fit2)
    fit2_diff = abs(prft_log_df[col_b].values - med_fit2)
    print(fit2_diff)
    prft_log_df['fit2_diff'] = fit2_diff
    med_prft_ind = prft_log_df.loc[(prft_log_df['fit2_diff'] == prft_log_df['fit2_diff'].min())]
    print("med_prft_ind", med_prft_ind)
    print(len(med_prft_ind))
    if len(med_prft_ind) > 1:
        med_prft_ind = med_prft_ind.loc[(med_prft_ind[col_b] == med_prft_ind[col_b].min())]
        print("med_prft_ind", med_prft_ind)



    mean_fit2 = np.mean(prft_log_df[col_b])
    print("med_fit2", mean_fit2)
    fit2_diff = abs(prft_log_df[col_b].values - mean_fit2)
    print(fit2_diff)
    prft_log_df['fit2_diff'] = fit2_diff
    mean_prft_ind = prft_log_df.loc[(prft_log_df['fit2_diff'] == prft_log_df['fit2_diff'].min())]
    print("mean_prft_ind", mean_prft_ind)
    print(len(mean_prft_ind))
    if len(mean_prft_ind) > 1:
        mean_prft_ind = mean_prft_ind.loc[(mean_prft_ind[col_b] == mean_prft_ind[col_b].min())]
        print("mean_prft_ind", mean_prft_ind)



    hof = []
    hof_1 = [min_prft_ind['p1'].values[0], min_prft_ind['p2'].values[0], min_prft_ind['p3'].values[0], min_prft_ind['p4'].values[0]]
    hof_2 = [med_prft_ind['p1'].values[0], med_prft_ind['p2'].values[0], med_prft_ind['p3'].values[0], med_prft_ind['p4'].values[0]]
    hof_3 = [mean_prft_ind['p1'].values[0], mean_prft_ind['p2'].values[0], mean_prft_ind['p3'].values[0], mean_prft_ind['p4'].values[0]]
    hof_1 = list(map(int, hof_1))
    hof_2 = list(map(int, hof_2))
    hof_3 = list(map(int, hof_3))
    hof.append(hof_1)
    hof.append(hof_2)
    hof.append(hof_3)



    print ("hof_1",hof_1)
    print ("hof_2", hof_2)
    print("hof_3", hof_3)
    print ("hof", hof)


    """ Creates a new instance of the training-validation task and computes the fitness of the current individual """

    for i in range(len(hof)):

        if i == 0:
            hof_ref = "min"
        else:
            hof_ref = "med"

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
        print("dtype(num_neuron_lst[0])", type(num_neuron_lst[0]))
        print("HoF type_neuron_lst: ", type_neuron_lst)

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

        output_lst = []
        truth_lst = []

        start = time.time()
        # Test
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



        for idx in range(len(units_index_test)):
            print(output_lst[idx])
            print(truth_lst[idx])
            fig_verify = plt.figure(figsize=(24, 10))
            plt.plot(output_lst[idx], color="green")
            plt.plot(truth_lst[idx], color="red", linewidth=2.0)
            plt.title('Unit%s inference' %str(int(units_index_test[idx])), fontsize=30)
            plt.yticks(fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel('RUL', fontdict={'fontsize': 24})
            plt.xlabel('Timestamps', fontdict={'fontsize': 24})
            plt.legend(['Predicted', 'Truth'], loc='upper right', fontsize=28)
            plt.ylim([-20, 80])
            plt.show()
            fig_verify.savefig(pic_dir + "/moo_elm_%s_unit_test_%s_pop%s_gen%s_rmse-%s_score-%s_time-%s.png" %(hof_ref,
                                                                                                       str(int(units_index_test[idx])),
                                                                                  str(args.pop), str(args.gen), str(rms), str(score), str(train_time)))

        end = time.time()

        test_time = end - start

        print("BEST model training time: ", train_time)
        print("BEST model testtime: ", test_time)
        print("HOF phenotype: ", [l2_parms_lst[hof[i][0] - 1], hof[i][1] * 10, hof[i][2] * 10, hof[i][3]])
        print(" test RMSE: ", rms)
        print(" test Score: ", score)




if __name__ == '__main__':
    main()
