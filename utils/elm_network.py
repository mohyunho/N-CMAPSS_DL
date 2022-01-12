import time
import json
import logging as log
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

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

from math import sqrt
# import keras
np.random.seed(0)

from utils.hpelm import ELM, HPELM


# def score_calculator(y_predicted, y_actual):
#     # Score metric
#     h_array = y_predicted - y_actual
#     s_array = np.zeros(len(h_array))
#     print ("calculating score")
#     for j, h_j in enumerate(h_array):
#         if h_j < 0:
#             s_array[j] = math.exp(-(h_j / 13)) - 1
#
#         else:
#             s_array[j] = math.exp(h_j / 10) - 1
#     score = np.sum(s_array)
#     return score


def gen_net(feat_len, l2_norm, lin_check, num_neurons_lst, type_lst, batch, device = "GPU"):
    '''
    Generate and evaluate any ELM
    :param
    :return:
    '''

    model = HPELM(feat_len, 1, accelerator=device, batch=batch, norm=l2_norm)
    for idx in range(2):
        # print ("idx", idx)
        # print ("num_neurons_lst[idx]", num_neurons_lst[idx])
        # print ("type_lst[idx]", type_lst[idx])
        model.add_neurons(num_neurons_lst[idx], type_lst[idx])

    if lin_check == 1:
        model.add_neurons(num_neurons_lst[2], type_lst[2])
    else:
        pass

    return model


class network_fit(object):
    '''
    class for network
    '''

    def __init__(self, feat_len,
                 l2_parm, lin_check, num_neurons_lst, type_lst, model_path, device, batch):
        '''
        Constructor
        Generate a NN and train
        @param none
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.feat_len = feat_len
        self.l2_parm = l2_parm
        self.lin_check = lin_check
        self.num_neurons_lst = num_neurons_lst
        self.type_lst = type_lst
        self.model_path = model_path
        self.device = device
        self.batch = batch


        # self.model= gen_net(self.feat_len, self.l2_parm, self.lin_check,
        #                     self.num_neurons_lst, self.type_lst, self.device)



    def train_net(self, model, train_sample_array, train_label_array, val_sample_array, val_label_array):
        '''
        specify the optimizers and train the network
        :param epochs:
        :param batch_size:
        :param lr:
        :return:
        '''
        print("Initializing network...")
        start_itr = time.time()
        elm = model
        elm.train(train_sample_array, train_label_array, "R")
        print ("individual trained...evaluation in progress...")

        neurons_lst, norm_check = elm.summary()
        print ("summary: ", neurons_lst, norm_check)

        pred_test = elm.predict(val_sample_array)
        pred_test = pred_test.flatten()
        # print ("pred_test.shape", pred_test.shape)
        # print ("self.val_label_array.shape", self.val_label_array.shape)

        # score = score_calculator(pred_test, self.val_label_array)
        # print("score: ", score)

        rms = sqrt(mean_squared_error(pred_test, val_label_array))
        # print(rms)
        rms = round(rms, 8)
        val_net = (rms,)
        end_itr = time.time()
        print("training network is successfully completed, time: ", end_itr - start_itr)
        print("val_net in rmse: ", val_net[0])

        elm = None
        pred_test = None
        del elm, pred_test


        return val_net


    def trained_model(self):
        model = gen_net(self.feat_len, self.l2_parm, self.lin_check,
                             self.num_neurons_lst, self.type_lst, self.batch, self.device)
        return model
