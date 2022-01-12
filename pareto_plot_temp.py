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

import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
# import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt





from utils.pareto import pareto
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

pop_size = 4
n_generations = 5

current_dir = os.path.dirname(os.path.abspath(__file__))

pic_dir = os.path.join(current_dir, 'Figures')
# Log file path of EA in csv
ea_log_path = os.path.join(current_dir, 'EA_log')

scale = 100

def roundup(x, scale):
    return int(math.ceil(x / float(scale))) * scale

def rounddown(x, scale):
    return int(math.floor(x / float(scale))) * scale


col_a = 'fitness_1'
col_b = 'fitness_2'
pd.options.mode.chained_assignment = None  # default='warn'

mute_log_file_path = os.path.join(ea_log_path, 'mute_log_%s_%s.csv' % (pop_size, n_generations))
# ea_log_path + 'mute_log_%s_%s.csv' % (pop_size, n_generations)
mute_log_df = pd.read_csv(mute_log_file_path)
print (mute_log_df)


prft_log_file_path = os.path.join(ea_log_path, 'prft_out_%s_%s.csv' % (pop_size, n_generations))
# ea_log_path + 'mute_log_%s_%s.csv' % (pop_size, n_generations)
prft_log_df = pd.read_csv(prft_log_file_path, header=0, names=["p1",'p2','p3','p4'])
# prft_log_df = pd.read_csv(prft_log_file_path)
print (prft_log_df)

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

    print ("log_prft_ind", log_prft_ind)
    fit1_lst.append(log_prft_ind[col_a].values[0])
    fit2_lst.append(log_prft_ind[col_b].values[0])

print ("fit1_lst", fit1_lst)
print ("fit2_lst", fit2_lst)

prft_log_df[col_a] = fit1_lst
prft_log_df[col_b] = fit2_lst

print ("prft_log_df", prft_log_df)


# solutions_df = mute_log_df[['val_rmse', 'penalty']]
# solutions_df['penalty'] = solutions_df['penalty'] * 1e4
# print (solutions_df)

# solutions_df = mute_log_df[['fitness_1', 'fitness_2']]

min_fit1 = np.min(prft_log_df[col_a])
print ("min_fit1", min_fit1)

min_prft_ind = prft_log_df.loc[(prft_log_df[col_a] == min_fit1)]
print ("min_prft_ind", min_prft_ind)

med_fit2 = np.median(prft_log_df[col_b])
print ("med_fit2", med_fit2)

fit2_diff = abs(prft_log_df[col_b].values - med_fit2)
print (fit2_diff)

prft_log_df['fit2_diff'] = fit2_diff



med_prft_ind = prft_log_df.loc[(prft_log_df['fit2_diff'] == prft_log_df['fit2_diff'].min())]
print ("med_prft_ind", med_prft_ind)
print (len(med_prft_ind))

if len(med_prft_ind) > 1:
    med_prft_ind = med_prft_ind.loc[(med_prft_ind[col_b] == med_prft_ind[col_b].min())]

print ("med_prft_ind", med_prft_ind)

print ("med_prft_ind['p1']", type(med_prft_ind['p1'].values[0]))
print ("med_prft_ind['p2']", med_prft_ind['p2'].values[0])
#
#
#
#
#
#
# #############################à
# data = solutions_df
#
#
# sets = {}
# archives = {}
#
# fig = matplotlib.figure.Figure(figsize=(15, 15))
# agg.FigureCanvasAgg(fig)
#
#
# # print ("data", data)
# # print ("columns", data.columns)
# # print ("data.itertuples(False)", data.itertuples(False))
# resolution = 1e-4
#
# archives = pareto.eps_sort([data.itertuples(False)], [0, 1], [resolution] * 2)
# sets = pd.DataFrame(data=archives.archive)
# # print ("archives", archives)
# # print ("sets", sets)
#
# spacing_x = 0.2
# spacing_y = 500
#
# fig = matplotlib.figure.Figure(figsize=(8, 8))
# agg.FigureCanvasAgg(fig)
#
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)
# ax.scatter(sets[col_a], sets[col_b], facecolor=(1.0, 1.0, 0.4), edgecolors =(0.0, 0.0, 0.0), zorder=1, s=50)
#
# x_max = round(max(data[col_a]), 1)
# y_max = roundup(max(data[col_b])+200,100)
#
# for box in archives.boxes:
#     ll = [box[0] * resolution, box[1] * resolution]
#
#     # make a rectangle in the Y direction
#     # rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), y_max - ll[0], y_max - ll[1], lw=1,
#     #                                     facecolor=(1.0, 0.8, 0.8), edgecolor=  (0.0,0.0,0.0), zorder=-10)
#     rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), y_max - ll[0], y_max - ll[1], lw=1,
#                                         facecolor=(1.0, 0.8, 0.8), zorder=-10)
#     ax.add_patch(rect)
#
#     # make a rectangle in the X direction
#     # rect = matplotlib.patches.Rectangle((ll[0] + resolution, ll[1]), x_max - ll[0], x_max - ll[1], lw=0,
#     #                                     facecolor=(1.0, 0.8, 0.8), zorder=-10)
#     ax.add_patch(rect)
# if resolution < 1e-3:
#     spacing = 0.1
# else:
#     spacing = resolution
#     while spacing < 0.2:
#         spacing *= 2
#
# ax.set_xticks(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x))
# ax.set_yticks(np.arange(rounddown(min(data[col_b])-100,scale), roundup(max(data[col_b])+100,scale), spacing_y))
#
# # if resolution > 0.001:
# #     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
# #     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
# ax.set_xlim(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2)
# ax.set_ylim(rounddown(min(data[col_b])-200, 100), roundup(max(data[col_b])+200,100))
# ax.set_title("Solutions and pareto front", fontsize=15)
# ax.set_xlabel('Validation RMSE', fontsize=13)
# ax.set_ylabel('Trainable parameters', fontsize=13)
#
# fig.savefig("example")
#
# fig = matplotlib.figure.Figure(figsize=(5, 5))
# agg.FigureCanvasAgg(fig)
#
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1)
#
# ax.set_xticks(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x))
# ax.set_yticks(np.arange(rounddown(min(data[col_b])-100,scale), roundup(max(data[col_b])+100,scale), spacing_y))
# ax.set_xlim(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2)
# ax.set_ylim(rounddown(min(data[col_b])-200, 100), roundup(max(data[col_b])+200,100))
# ax.set_title("Unsorted Data")
# ax.set_xlabel(r'$f_1$')
# ax.set_ylabel(r'$f_2$')
#
# fig.savefig("unsorted")
#
# ####################
# hv_lst = []
# for gen in mute_log_df['gen'].unique():
#   hv_temp = mute_log_df.loc[mute_log_df['gen']==gen]['hypervolume'].values
#   hv_value = sum(hv_temp)/len(hv_temp)
#   hv_lst.append(hv_value)
#
# print (hv_lst)
#
# norm_hv = [x / (max(hv_lst)+1) for x in hv_lst]
# print (norm_hv)
# x_ref = range(1,n_generations+1)
#
# fig_verify = plt.figure(figsize=(8, 5))
# plt.plot(x_ref, norm_hv)
# plt.xticks(x_ref)
# plt.ylabel("Normalized hypervolume", fontsize=13)
# plt.xlabel("Generations", fontsize=13)
# fig_verify.savefig(os.path.join(ea_log_path, 'hv_plot.png'))
#
