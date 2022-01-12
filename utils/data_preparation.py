import gc

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

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

def df_all_creator(data_filepath):
    """

     """
    # Time tracking, Operation time (min):  0.003
    t = time.process_time()


    with h5py.File(data_filepath, 'r') as hdf:
        # Development(training) set
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
        T_dev = np.array(hdf.get('T_dev'))  # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))


    W = np.concatenate((W_dev, W_test), axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T = np.concatenate((T_dev, T_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0)

    print('')
    print("Operation time (min): ", (time.process_time() - t) / 60)
    print("number of training samples(timestamps): ", Y_dev.shape[0])
    print("number of training samples(timestamps): ", Y_test.shape[0])
    print('')
    print("W shape: " + str(W.shape))
    print("X_s shape: " + str(X_s.shape))
    print("X_v shape: " + str(X_v.shape))
    print("T shape: " + str(T.shape))
    print("Y shape: " + str(Y.shape))
    print("A shape: " + str(A.shape))

    '''
    Illusration of Multivariate time-series of condition monitoring sensors readings for Unit5 (fifth engine)

    W: operative conditions (Scenario descriptors) - ['alt', 'Mach', 'TRA', 'T2']
    X_s: measured signals - ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
    X_v: virtual sensors - ['T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']
    T(theta): engine health parameters - ['fan_eff_mod', 'fan_flow_mod', 'LPC_eff_mod', 'LPC_flow_mod', 'HPC_eff_mod', 'HPC_flow_mod', 'HPT_eff_mod', 'HPT_flow_mod', 'LPT_eff_mod', 'LPT_flow_mod']
    Y: RUL [in cycles]
    A: auxiliary data - ['unit', 'cycle', 'Fc', 'hs']
    '''

    df_W = pd.DataFrame(data=W, columns=W_var)
    df_Xs = pd.DataFrame(data=X_s, columns=X_s_var)
    df_Xv = pd.DataFrame(data=X_v[:,0:2], columns=['T40', 'P30'])
    # df_T = pd.DataFrame(data=T, columns=T_var)
    df_Y = pd.DataFrame(data=Y, columns=['RUL'])
    df_A = pd.DataFrame(data=A, columns=A_var).drop(columns=['cycle', 'Fc', 'hs'])



    # Merge all the dataframes
    df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1)

    print (df_all)    # df_all = pd.concat([df_W, df_Xs, df_Xv, df_Y, df_A], axis=1).drop(columns=[ 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi', 'Fc', 'hs'])

    # del [[df_W, df_Xs, df_Xv, df_Y, df_A]]
    # gc.collect()
    # df_W = pd.DataFrame()
    # df_Xs = pd.DataFrame()
    # df_Xv = pd.DataFrame()
    # df_Y = pd.DataFrame()
    # df_A = pd.DataFrame()

    return df_all



def df_train_creator(df_all, units_index_train):
    train_df_lst= []
    for idx in units_index_train:
        df_train_temp = df_all[df_all['unit'] == np.float64(idx)]
        train_df_lst.append(df_train_temp)
    df_train = pd.concat(train_df_lst)
    df_train = df_train.reset_index(drop=True)

    return df_train


def df_test_creator(df_all, units_index_test):
    test_df_lst = []
    for idx in units_index_test:
        df_test_temp = df_all[df_all['unit'] == np.float64(idx)]
        test_df_lst.append(df_test_temp)

    df_test = pd.concat(test_df_lst)
    df_test = df_test.reset_index(drop=True)

    return df_test

def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 142 192 -> from row 142 to 192
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

def time_window_slicing (input_array, sequence_length, sequence_cols):
    # generate labels
    label_gen = [gen_labels(input_array[input_array['unit'] == id], sequence_length, ['RUL'])
                 for id in input_array['unit'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)
    # label_array = np.concatenate(label_gen)

    # transform each id of the train dataset in a sequence
    seq_gen = (list(gen_sequence(input_array[input_array['unit'] == id], sequence_length, sequence_cols))
               for id in input_array['unit'].unique())
    sample_array = np.concatenate(list(seq_gen)).astype(np.float32)
    # sample_array = np.concatenate(list(seq_gen))

    print("sample_array")
    return sample_array, label_array


def time_window_slicing_label_save (input_array, sequence_length, stride, index, sample_dir_path, sequence_cols = 'RUL'):
    '''
    ref
        for i in range(0, input_temp.shape[0] - sequence_length):
        window = input_temp[i*stride:i*stride + sequence_length, :]  # each individual window
        window_lst.append(window)
        # print (window.shape)


    '''
    # generate labels
    window_lst = []  # a python list to hold the windows

    input_temp = input_array[input_array['unit'] == index][sequence_cols].values
    num_samples = int((input_temp.shape[0] - sequence_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + sequence_length]  # each individual window
        window_lst.append(window)
        # print (window.shape)

    label_array = np.asarray(window_lst).astype(np.float32)
    # label_array = np.asarray(window_lst)


    # np.save(os.path.join(sample_dir_path, 'Unit%s_rul_win%s_str%s' %(str(int(index)), sequence_length, stride)),
    #         label_array)  # save the file as "outfile_name.npy"

    return label_array[:,-1]



def time_window_slicing_sample_save (input_array, sequence_length, stride, index, sample_dir_path, sequence_cols):
    '''


    '''
    # generate labels
    window_lst = []  # a python list to hold the windows

    input_temp = input_array[input_array['unit'] == index][sequence_cols].values
    print ("Unit%s input array shape: " %index, input_temp.shape)
    num_samples = int((input_temp.shape[0] - sequence_length)/stride) + 1
    for i in range(num_samples):
        window = input_temp[i*stride:i*stride + sequence_length,:]  # each individual window
        window_lst.append(window)

    sample_array = np.dstack(window_lst).astype(np.float32)
    # sample_array = np.dstack(window_lst)
    print ("sample_array.shape", sample_array.shape)

    # np.save(os.path.join(sample_dir_path, 'Unit%s_samples_win%s_str%s' %(str(int(index)), sequence_length, stride)),
    #         sample_array)  # save the file as "outfile_name.npy"


    return sample_array



class Input_Gen(object):
    '''
    class for data preparation (sequence generator)
    '''

    def __init__(self, df_train, df_test, cols_normalize, sequence_length, sequence_cols, sample_dir_path,
                 unit_index, stride):
        '''

        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        print("the number of input signals: ", len(cols_normalize))
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        norm_df = pd.DataFrame(min_max_scaler.fit_transform(df_train[cols_normalize]),
                               columns=cols_normalize,
                               index=df_train.index)
        join_df = df_train[df_train.columns.difference(cols_normalize)].join(norm_df)
        df_train = join_df.reindex(columns=df_train.columns)

        norm_test_df = pd.DataFrame(min_max_scaler.transform(df_test[cols_normalize]), columns=cols_normalize,
                                    index=df_test.index)
        test_join_df = df_test[df_test.columns.difference(cols_normalize)].join(norm_test_df)
        df_test = test_join_df.reindex(columns=df_test.columns)
        df_test = df_test.reset_index(drop=True)

        self.df_train = df_train
        self.df_test = df_test

        print (self.df_train)
        print (self.df_test)

        self.cols_normalize = cols_normalize
        self.sequence_length = sequence_length
        self.sequence_cols = sequence_cols
        self.sample_dir_path = sample_dir_path
        self.unit_index = np.float64(unit_index)
        self.stride = stride


    def seq_gen(self):
        '''
        concatenate vectors for NNs
        :param :
        :param :
        :return:
        '''

        if any(index == self.unit_index for index in self.df_train['unit'].unique()):
            print ("Unit for Train")
            label_array = time_window_slicing_label_save(self.df_train, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols='RUL')
            sample_array = time_window_slicing_sample_save(self.df_train, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols=self.cols_normalize)

        else:
            print("Unit for Test")
            label_array = time_window_slicing_label_save(self.df_test, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols='RUL')
            sample_array = time_window_slicing_sample_save(self.df_test, self.sequence_length,
                                           self.stride, self.unit_index, self.sample_dir_path, sequence_cols=self.cols_normalize)

        sample_split_lst = np.array_split(sample_array, 3, axis=2)
        # print (sample_split_lst[0].shape)
        # print(sample_split_lst[1].shape)
        # print(sample_split_lst[2].shape)

        label_split_lst = np.array_split(label_array, 3, axis=0)
        # print (label_split_lst[0].shape)
        # print(label_split_lst[1].shape)
        # print(label_split_lst[2].shape)



        np.savez_compressed(os.path.join(self.sample_dir_path, 'Unit%s_win%s_str%s_part1' %(str(int(self.unit_index)), self.sequence_length, self.stride)),
                                         sample=sample_split_lst[0], label=label_split_lst[0])
        print ("part1 saved")
        np.savez_compressed(os.path.join(self.sample_dir_path, 'Unit%s_win%s_str%s_part2' %(str(int(self.unit_index)), self.sequence_length, self.stride)),
                                         sample=sample_split_lst[1], label=label_split_lst[1])
        print ("part2 saved")
        np.savez_compressed(os.path.join(self.sample_dir_path, 'Unit%s_win%s_str%s_part3' %(str(int(self.unit_index)), self.sequence_length, self.stride)),
                                         sample=sample_split_lst[2], label=label_split_lst[2])
        print ("part3 saved")


        return




