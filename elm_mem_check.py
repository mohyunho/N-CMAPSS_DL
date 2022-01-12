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
from utils.ea import GeneticAlgorithm
import gc

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


# Log file path of EA in csv
directory_path = current_dir + '/EA_log'

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

def figsave(history, h1,h2,h3,h4, bs, lr, sub):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/elm_enas_training_h1%s_h2%s_h3%s_h4%s_bs%s_sub%s_lr%s.png" %(int(h1), int(h2), int(h3), int(h4), int(bs), int(sub), str(lr)))
    return


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


def recursive_clean(directory_path):
    """clean the whole content of :directory_path:"""
    if os.path.isdir(directory_path) and os.path.exists(directory_path):
        files = glob.glob(directory_path + '*')
        for file_ in files:
            if os.path.isdir(file_):
                recursive_clean(file_ + '/')
            else:
                os.remove(file_)

units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
units_index_test = [11.0, 14.0, 15.0]



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='RPs creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-bs', type=int, default=1000, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=20, help='patience')
    parser.add_argument('-vs', type=float, default=0.1, help='validation split')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')




    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="GPU", help='Use "basic" if GPU with cuda is not available')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    lr = args.lr
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

    sample_array = []
    label_array = []

    ## Parameters for the GA
    pop_size = args.pop
    n_generations = args.gen
    cx_prob = 0.5  # 0.25
    mut_prob = 0.5  # 0.7
    cx_op = "one_point"
    mut_op = "uniform"
    sel_op = "best"
    other_args = {
        'mut_gene_probability': 0.3  # 0.1
    }



    start = time.time()

    # log_file_path = log_path + 'log_%s_%s_pop-%s_gen-%s_%s.csv' % (
    # subdata_mode_list[subdata_mode], fitness_mode_list[fitness_mode], pop_size, n_generations, trial)
    # log_col = ['idx', 'stop_epoch', 'window_length', 'n_filters', 'kernel_size', 'n_conv_layer', 'LSTM1_units',
    #            'LSTM2_units', 'n_window',
    #            'val_rmse', 'val_score', 'rmse_combined', 'score_combined',
    #            'AIC', 'train_loss', 'mle_term', 'params_term', 'geno_list']
    # log_df = pd.DataFrame(columns=log_col, index=None)
    # log_df.to_csv(log_file_path, index=False)
    # print(log_df)


    # Save log file of EA in csv
    recursive_clean(directory_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    mutate_log_path = 'EA_log/mute_log_%s_%s.csv' % (pop_size, n_generations)
    mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'params_5', 'params_6', 'fitness', 'gen']
    mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
    mutate_log_df.to_csv(mutate_log_path, index=False)




    def log_function(population, gen, mutate_log_path = mutate_log_path):
        for i in range(len(population)):
            if population[i] == []:
                "non_mutated empty"
                pass
            else:
                # print ("i: ", i)
                population[i].append(population[i].fitness.values[0])
                population[i].append(gen)

        temp_df = pd.DataFrame(np.array(population), index=None)
        temp_df.to_csv(mutate_log_path, mode='a', header=None)
        print("population saved")
        return





    """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
    def eval_loop_class(train_sample_array,train_label_array, val_sample_array , val_label_array):
        l2_parms_lst = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        l2_parm = l2_parms_lst[3 - 1]
        type_neuron_lst = ["tanh", "sigm", "rbf_l2", "rbf_linf", "lin"]

        lin_check = 1
        num_neuron_lst = []
        hof = [[10,10,10,10,10,10,10,10,10,10,10,10], [10,10,10,10,10,10,10,10,10,10,10,10]]
        for n in range(4):
            num_neuron_lst.append(hof[0][n + 1] * 10)
        if lin_check == 1:
            num_neuron_lst.append(20)
        else:
            num_neuron_lst.append(0)

        print("HoF l2_params: ", l2_parm)
        print("HoF lin_check: ", lin_check)
        print("HoF num_neuron_lst: ", num_neuron_lst)
        print("HoF type_neuron_lst: ", type_neuron_lst)

        feat_len = train_sample_array.shape[1]
        best_elm_class = network_fit(feat_len,
                              l2_parm, lin_check,
                              num_neuron_lst, type_neuron_lst, model_temp_path, device)
        best_elm_net = best_elm_class.trained_model()

        # Train the best network
        sample_array = np.concatenate((train_sample_array, val_sample_array))
        label_array = np.concatenate((train_label_array, val_label_array))

        print ("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        best_elm_net.train(train_sample_array, train_label_array, "R")
        print("individual trained...evaluation in progress...")
        neurons_lst, norm_check = best_elm_net.summary()
        print("summary: ", neurons_lst, norm_check)


        pred_test = best_elm_net.predict(val_sample_array)
        pred_test = pred_test.flatten()
        # print ("pred_test.shape", pred_test.shape)
        # print ("self.val_label_array.shape", self.val_label_array.shape)

        # score = score_calculator(pred_test, self.val_label_array)
        # print("score: ", score)

        rms = sqrt(mean_squared_error(pred_test, val_label_array))
        # print(rms)
        rms = round(rms, 4)

        best_elm_class = []
        best_elm_net = []
        sample_array = []
        label_array  = []
        train_sample_array  = []
        train_label_array  = []
        val_sample_array  = []
        val_label_array  = []
        pred_test  = []

        del best_elm_class, best_elm_net, sample_array, label_array, train_sample_array, train_label_array
        del val_sample_array, val_label_array, pred_test
        gc.collect()
        return rms


    def eval_loop(train_sample_array,train_label_array, val_sample_array , val_label_array):
        l2_parms_lst = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        l2_parm = l2_parms_lst[3 - 1]
        type_neuron_lst = ["tanh", "sigm", "tanh", "sigm", "lin"]

        lin_check = 1
        num_neuron_lst = []
        hof = [[10,10,10,10,10,10,10,10,10,10,10,10], [10,10,10,10,10,10,10,10,10,10,10,10]]
        for n in range(4):
            num_neuron_lst.append(hof[0][n + 1] * 10)
        if lin_check == 1:
            num_neuron_lst.append(20)
        else:
            num_neuron_lst.append(0)

        print("HoF l2_params: ", l2_parm)
        print("HoF lin_check: ", lin_check)
        print("HoF num_neuron_lst: ", num_neuron_lst)
        print("HoF type_neuron_lst: ", type_neuron_lst)

        feat_len = train_sample_array.shape[1]
        model = HPELM(feat_len, 1, accelerator=device, batch=5000, norm=l2_parm)
        for idx in range(4):
            model.add_neurons(num_neuron_lst[idx], type_neuron_lst[idx])
        if lin_check == 1:
            model.add_neurons(num_neuron_lst[4], type_neuron_lst[4])
        else:
            pass


        # Train the best network
        sample_array = np.concatenate((train_sample_array, val_sample_array))
        label_array = np.concatenate((train_label_array, val_label_array))

        print ("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        model.train(train_sample_array, train_label_array, "R")
        print("individual trained...evaluation in progress...")
        neurons_lst, norm_check = model.summary()
        print("summary: ", neurons_lst, norm_check)


        pred_test = model.predict(val_sample_array)
        pred_test = pred_test.flatten()
        # print ("pred_test.shape", pred_test.shape)
        # print ("self.val_label_array.shape", self.val_label_array.shape)

        # score = score_calculator(pred_test, self.val_label_array)
        # print("score: ", score)



        rms = sqrt(mean_squared_error(pred_test, val_label_array))
        # print(rms)
        rms = round(rms, 4)

        model.net_reset()

        best_elm_class = []
        best_elm_net = []
        sample_array = []
        label_array  = []
        train_sample_array  = []
        train_label_array  = []
        val_sample_array  = []
        val_label_array  = []
        pred_test  = []
        model = []

        del best_elm_class, best_elm_net, sample_array, label_array, train_sample_array, train_label_array
        del val_sample_array, val_label_array, pred_test
        del model
        gc.collect()
        return rms



    rms_lst = []
    # for i in range(10):
    #     print ("i", i)
    #     rms = eval_loop(train_sample_array,train_label_array, val_sample_array , val_label_array)
    #     rms_lst.append(rms)

    print ("1")
    rms = eval_loop(train_sample_array,train_label_array, val_sample_array , val_label_array)
    rms_lst.append(rms)
    print(dir())
    print("2")
    rms = eval_loop(train_sample_array,train_label_array, val_sample_array , val_label_array)
    rms_lst.append(rms)
    print(dir())
    print("3")
    rms = eval_loop(train_sample_array,train_label_array, val_sample_array , val_label_array)
    rms_lst.append(rms)
    print(dir())
    print("4")
    rms = eval_loop(train_sample_array,train_label_array, val_sample_array , val_label_array)
    rms_lst.append(rms)
    print(dir())







if __name__ == '__main__':
    main()
