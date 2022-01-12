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
from random import shuffle
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

seed = 0
random.seed(0)
np.random.seed(seed)

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

from utils.data_preparation_unit import df_all_creator, df_train_creator, df_test_creator, Input_Gen
from utils.dnn import one_dcnn, CNNBranch, TD_CNNBranch, CNNB, multi_head_cnn, sensor_input_model


initializer = GlorotNormal(seed=0)

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel(logging.ERROR)



# tf.config.set_visible_devices([], 'GPU')

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnnlstm_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')

pic_dir = os.path.join(current_dir, 'Figures')

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

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

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

def figsave(history, win_len, win_stride, bs, sub, lr, sub_win_stride, sub_win_len, mul1, mul2):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/cnnlstm_training_w%s_s%s_bs%s_sub%s_lr%s_substride%s_subwinlen%s_mul1-%s_mul2%s.png" %(int(win_len), int(win_stride), int(bs),
                                                                                int(sub), str(lr), str(sub_win_stride), str(sub_win_len), str(mul1), str(mul2)))
    return


def segment_gen(seq_array_train, seg_n, sub_win_stride, sub_win_len):
    '''
    ## Reshape the time series as the network input
    # for each sensor: reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
    '''

    train_FD_sensor = []

    as_strided = np.lib.stride_tricks.as_strided

    for s_i in range(seq_array_train.shape[2]):
        window_list = []
        window_array = np.array([])

        for seq in range(seq_array_train.shape[0]):
            S = sub_win_stride
            s0 = seq_array_train[seq, :, s_i].strides
            seq_sensor = as_strided(seq_array_train[seq, :, s_i], (seg_n, sub_win_len), strides=(S * s0[0], s0[0]))
            #         print (seq_sensor)
            #         window_array = np.concatenate((window_array, seq_sensor), axis=1)
            window_list.append(seq_sensor)

        window_array = np.stack(window_list, axis=0)
        window_array = np.reshape(window_array, (window_array.shape[0], window_array.shape[1], window_array.shape[2], 1))
        print(str(s_i) + "-" + str(window_array.shape))
        train_FD_sensor.append(window_array)

    # print("train_FD_sensor[0].shape", train_FD_sensor[0].shape)
    # print(seq_array_train[0, :, 0])
    return train_FD_sensor



def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops



def scheduler(epoch, lr):
    if epoch == 10:
        print("lr decay by 10")
        return lr * 0.1
    elif epoch == 30:
        print("lr decay by 10")
        return lr * 0.1
    else:
        return lr



def release_list(a):
   del a[:]
   del a




units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
#units_index_train = [20.0, 18.0, 16.0, 10.0, 5.0, 2.0]
units_index_test = [11.0, 14.0, 15.0]

sensor_col = ['alt', 'mach', 'tra', 't2', 't24', 't30', 't48', 't50', 'p15', 'p2',
       'p21', 'p24', 'ps30', 'p40', 'p50', 'nf', 'nc', 'wf', 't40', 'p30']



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-f', type=int, default=10, help='number of filter')
    parser.add_argument('-k', type=int, default=10, help='size of kernel')
    parser.add_argument('-bs', type=int, default=256, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=20, help='patience')
    parser.add_argument('-vs', type=float, default=0.1, help='validation split')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=10, help='subsampling stride')
    parser.add_argument('-s_stride', type=int, default=10, help='subwindow stride')
    parser.add_argument('-s_len', type=int, default=100, help='subwindow len')
    parser.add_argument('-n_conv', type=int, default=3, help='number of conv layers')
    parser.add_argument('-lstm1', type=int, default=10, help='lstm1 unit multiplier')
    parser.add_argument('-lstm2', type=int, default=5, help='lstm2 unit multiplier')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    n_filters = args.f
    kernel_size = args.k
    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub


    input_features = 1
    sub_win_stride = args.s_stride
    sub_win_len = args.s_len
    seg_n = int((win_len - sub_win_len) / (sub_win_stride) + 1)
    n_conv_layer = args.n_conv

    print("the number of segments:", seg_n)

    bidirec = False

    mul1 = args.lstm1
    mul2 = args.lstm2
    LSTM_u1 = seg_n*mul1
    LSTM_u2 = seg_n*mul2

    # LSTM_u1 = args.lstm1
    # LSTM_u2 = args.lstm2

    n_outputs = 1

    amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
    rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                               name='RMSprop')


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

    # sample_array, label_array = shuffle_array(sample_array, label_array)
    print("samples are shuffled")
    print("sample_array.shape", sample_array.shape)
    print("label_array.shape", label_array.shape)

    # Convert  (#samples, win_len) of each sensor to (#samples. subseq, sub_win_len)
    train_FD_sensor = segment_gen(sample_array, seg_n, sub_win_stride, sub_win_len)


    sensor_input_shape = sensor_input_model(sensor_col, seg_n, sub_win_len, input_features)
    cnn_out_list, cnn_branch_list = multi_head_cnn(sensor_input_shape, n_filters, sub_win_len,
                                                   seg_n, input_features, sub_win_stride, kernel_size,
                                                   n_conv_layer, initializer)

    # x = concatenate([cnn_1_out, cnn_2_out])
    x = concatenate(cnn_out_list)
    # x = keras.layers.concatenate([lstm_out, auxiliary_input])

    # We stack a deep densely-connected network on top
    if bidirec == True:
        x = Bidirectional(LSTM(units=LSTM_u1, return_sequences=True, kernel_initializer=initializer))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(units=LSTM_u2, return_sequences=False, kernel_initializer=initializer))(x)
        x = Dropout(0.2)(x)
    elif bidirec == False:
        x = LSTM(units=LSTM_u1, return_sequences=True, kernel_initializer=initializer)(x)
        x = Dropout(0.2)(x)
        x = LSTM(units=LSTM_u2, return_sequences=False, kernel_initializer=initializer)(x)
        x = Dropout(0.2)(x)


    main_output = Dense(n_outputs, activation='linear', name='main_output', kernel_initializer=initializer)(x)
    cnnlstm = Model(inputs=sensor_input_shape, outputs=main_output)
    # model = Model(inputs=[input_1, input_2], outputs=main_output)


    start = time.time()

    lr_scheduler = LearningRateScheduler(scheduler)

    cnnlstm.compile(loss='mean_squared_error', optimizer=amsgrad,
                    metrics='mae')
    print(cnnlstm.summary())



    # fit the network
    history = cnnlstm.fit(train_FD_sensor, label_array, epochs=ep, batch_size=bs, validation_split=vs, verbose=2,
                          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=pt, verbose=1, mode='min'),
                                     ModelCheckpoint(model_temp_path, monitor='val_loss', save_best_only=True,
                                                     mode='min', verbose=1)])

    figsave(history, win_len, win_stride, bs, sub, lr, sub_win_stride, sub_win_len, mul1, mul2)

    print("The FLOPs is:{}".format(get_flops(cnnlstm)), flush=True)
    num_train = sample_array.shape[0]
    end = time.time()
    training_time = end - start
    print("Training time: ", training_time)

    ### Test (inference after training)
    start = time.time()

    output_lst = []
    truth_lst = []



    for index in units_index_test:
        print("Load data of: ", index)
        sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride)

        test_FD_sensor = segment_gen(sample_array, seg_n, sub_win_stride, sub_win_len)

        # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
        estimator = load_model(model_temp_path)

        y_pred_test = estimator.predict(test_FD_sensor)
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
    rms = sqrt(mean_squared_error(output_array, trytg_array))
    print(rms)
    rms = round(rms, 2)

    end = time.time()
    inference_time = end - start
    num_test = output_array.shape[0]

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
        plt.show()
        fig_verify.savefig(pic_dir + "/cnnlstm_unit%s_test_w%s_s%s_bs%s_lr%s_substride%s_subwinlen%s_mul1-%s_mul2%s_rms%s.png" %(str(int(units_index_test[idx])), int(win_len), int(win_stride), int(bs), str(lr), str(sub_win_stride), str(sub_win_len), str(mul1), str(mul2), str(rms)))


    print("The FLOPs is:{}".format(get_flops(cnnlstm)), flush=True)
    print("wind length_%s,  win stride_%s" %(str(win_len), str(win_stride)))
    print("# Training samples: ", num_train)
    print("# Inference samples: ", num_test)
    print("Training time: ", training_time)
    print("Inference time: ", inference_time)
    print("Result in RMSE: ", rms)


if __name__ == '__main__':
    main()

