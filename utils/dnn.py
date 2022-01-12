import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import random
from random import shuffle
from tqdm.keras import TqdmCallback

seed = 0
random.seed(0)
np.random.seed(seed)


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
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

initializer = GlorotNormal(seed=0)
#initializer = GlorotUniform(seed=0)


def one_dcnn(n_filters, kernel_size, input_array, initializer):

    cnn = Sequential(name='one_d_cnn')
    cnn.add(Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer=initializer, padding='same', input_shape=(input_array.shape[1],input_array.shape[2])))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv1D(filters=1, kernel_size=kernel_size, kernel_initializer=initializer, padding='same'))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Flatten())
    cnn.add(Dense(50, kernel_initializer=initializer))
    cnn.add(Activation('relu'))
    cnn.add(Dense(1, kernel_initializer=initializer))
    cnn.add(Activation("linear"))
    return cnn


'''
Define the function for generating CNN braches(heads)

'''


def CNNBranch(n_filters, window_length, input_features,
              strides_len, kernel_size, n_conv_layer):
    inputs = Input(shape=(window_length, input_features), name='input_layer')
    x = inputs
    for layer in range(n_conv_layer):
        x = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    outputs = Flatten()(x)
    #     causalcnn = Model(inputs, outputs=[outputs])
    cnnbranch = Model(inputs, outputs=outputs)
    return cnnbranch


def TD_CNNBranch(n_filters, window_length, n_window, input_features,
                 strides_len, kernel_size, n_conv_layer, initializer):
    cnn = Sequential()
    cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer),
                            input_shape=(n_window, window_length, input_features)))
    cnn.add(TimeDistributed(BatchNormalization()))
    cnn.add(TimeDistributed(Activation('relu')))

    if n_conv_layer == 1:
        pass

    elif n_conv_layer == 2:
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

    elif n_conv_layer == 3:
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

    elif n_conv_layer == 4:
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))
        cnn.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer=initializer)))
        cnn.add(TimeDistributed(BatchNormalization()))
        cnn.add(TimeDistributed(Activation('relu')))

    cnn.add(TimeDistributed(Flatten()))
    print(cnn.summary())

    return cnn


def CNNB(n_filters, lr, decay, loss,
         seq_len, input_features,
         strides_len, kernel_size,
         dilation_rates):
    inputs = Input(shape=(seq_len, input_features), name='input_layer')
    x = inputs
    for dilation_rate in dilation_rates:
        x = Conv1D(filters=n_filters,
                   kernel_size=kernel_size,
                   padding='causal',
                   dilation_rate=dilation_rate,
                   activation='linear')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # x = Dense(7, activation='relu', name='dense_layer')(x)
    outputs = Dense(3, activation='sigmoid', name='output_layer')(x)
    causalcnn = Model(inputs, outputs=[outputs])

    return causalcnn


def multi_head_cnn(sensor_input_model, n_filters, window_length, n_window,
                   input_features, strides_len, kernel_size, n_conv_layer, initializer):
    cnn_out_list = []
    cnn_branch_list = []

    for sensor_input in sensor_input_model:
        cnn_branch_temp = TD_CNNBranch(n_filters, window_length, n_window,
                                       input_features, strides_len, kernel_size, n_conv_layer, initializer)
        cnn_out_temp = cnn_branch_temp(sensor_input)

        cnn_branch_list.append(cnn_branch_temp)
        cnn_out_list.append(cnn_out_temp)

    return cnn_out_list, cnn_branch_list


def sensor_input_model(sensor_col, n_window, window_length, input_features):
    sensor_input_model = []
    for sensor in sensor_col:
        input_temp = Input(shape=(n_window, window_length, input_features), name='%s' % sensor)
        sensor_input_model.append(input_temp)

    return sensor_input_model



def cudnnlstm(sequence_length, nb_features, lstm1, lstm2, nb_out, initializer):
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=lstm1, kernel_initializer= initializer,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=lstm2, kernel_initializer= initializer,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))


    return model



def mlps(vec_len, h1, h2, h3, h4):
    '''

    '''
    model = Sequential()
    model.add(Dense(h1, activation='relu', input_shape=(vec_len,)))
    model.add(Dense(h2, activation='relu'))
    model.add(Dense(h3, activation='relu'))
    model.add(Dense(h4, activation='relu'))
    model.add(Dense(1))

    return model