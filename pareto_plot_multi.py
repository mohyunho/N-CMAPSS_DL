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
from itertools import cycle

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

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.pareto import pareto
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

pop_size = 28
n_generations = 30

current_dir = os.path.dirname(os.path.abspath(__file__))

pic_dir = os.path.join(current_dir, 'Figures')
# Log file path of EA in csv
ea_log_path = os.path.join(current_dir, 'EA_log')

scale = 100

def roundup(x, scale):
    return int(math.ceil(x / float(scale))) * scale

def rounddown(x, scale):
    return int(math.floor(x / float(scale))) * scale


pd.options.mode.chained_assignment = None  # default='warn'

results_lst = []
prft_lst = []
hv_trial_lst = []
prft_trial_lst = []
########################################
for file in sorted(os.listdir(ea_log_path)):
    if file.startswith("mute_log_28_30"):
        print ("path1: ", file)
        mute_log_df = pd.read_csv(os.path.join(ea_log_path, file))
        results_lst.append(mute_log_df)
    elif file.startswith("prft_out_28_30"):
        print("path2: ", file)
        prft_log_df = pd.read_csv(os.path.join(ea_log_path, file), header=0, names=["p1", 'p2', 'p3', 'p4'])
        prft_lst.append(prft_log_df)




for loop_idx in range(len(results_lst)):
    print ("file %s in progress..." %loop_idx)
    mute_log_df = results_lst[loop_idx]
    prft_log_df = prft_lst[loop_idx]

    col_a = 'fitness_1'
    col_b = 'fitness_2'
    solutions_df = mute_log_df[['fitness_1', 'fitness_2']]
    prft_trial_lst.append(prft_log_df)

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

        fit1_lst.append(log_prft_ind[col_a].values[0])
        fit2_lst.append(log_prft_ind[col_b].values[0])


    prft_log_df[col_a] = fit1_lst
    prft_log_df[col_b] = fit2_lst

#     #############################à
#     data = solutions_df
#
#     sets = {}
#     archives = {}
#
#     fig = matplotlib.figure.Figure(figsize=(15, 15))
#     agg.FigureCanvasAgg(fig)
#
#     # print ("data", data)
#     # print ("columns", data.columns)
#     # print ("data.itertuples(False)", data.itertuples(False))
#     resolution = 1e-4
#
#     archives = pareto.eps_sort([data.itertuples(False)], [0, 1], [resolution] * 2)
#     # print ("archives", archives)
#     # print ("sets", sets)
#
#     spacing_x = 0.5
#     spacing_y = 500
#
#     fig = matplotlib.figure.Figure(figsize=(6, 6))
#     agg.FigureCanvasAgg(fig)
#
#     ax = fig.add_subplot(1, 1, 1)
#     ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1, label="All solutions")
#     ax.scatter(prft_log_df[col_a], prft_log_df[col_b], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
#                s=50, label="Pareto front")
#
#     x_max = 13
#     y_max = 4000
#
#     for box in archives.boxes:
#         ll = [box[0] * resolution, box[1] * resolution]
#
#         # make a rectangle in the Y direction
#         # rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), y_max - ll[0], y_max - ll[1], lw=1,
#         #                                     facecolor=(1.0, 0.8, 0.8), edgecolor=  (0.0,0.0,0.0), zorder=-10)
#         rect = matplotlib.patches.Rectangle((ll[0], ll[1] + resolution), y_max - ll[0], y_max - ll[1], lw=1,
#                                             facecolor=(1.0, 0.8, 0.8), zorder=-10)
#         ax.add_patch(rect)
#
#         # make a rectangle in the X direction
#         # rect = matplotlib.patches.Rectangle((ll[0] + resolution, ll[1]), x_max - ll[0], x_max - ll[1], lw=0,
#         #                                     facecolor=(1.0, 0.8, 0.8), zorder=-10)
#         ax.add_patch(rect)
#     if resolution < 1e-3:
#         spacing = 0.1
#     else:
#         spacing = resolution
#         while spacing < 0.2:
#             spacing *= 2
#
#     x_range = np.arange(6, 13, spacing_x)
#     ax.set_xticks(x_range)
#     ax.set_xticklabels(x_range, rotation=60)
#     ax.set_yticks(
#         np.arange(0, 4000, spacing_y))
#     # ax.set_xticklabels(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x), rotation=60)
#     # if resolution > 0.001:
#     #     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
#     #     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
#     ax.set_xlim(6,13)
#     ax.set_ylim(-500,4000)
#     # ax.set_title("Solutions and pareto front", fontsize=15)
#     ax.set_xlabel('Validation RMSE', fontsize=15)
#     ax.set_ylabel('Trainable parameters', fontsize=15)
#     ax.legend(fontsize=11)
#     fig.savefig(os.path.join(pic_dir, 'prft_auto_%s_%s_t%s.png' % (pop_size, n_generations, loop_idx)), dpi=1500, bbox_inches='tight')
#     fig.savefig(os.path.join(pic_dir, 'prft_auto_%s_%s_t%s.eps' % (pop_size, n_generations, loop_idx)), dpi=1500, bbox_inches='tight')
#
#     #############################à
#
#     ####################
#     hv_lst = []
#     for gen in mute_log_df['gen'].unique():
#         hv_temp = mute_log_df.loc[mute_log_df['gen'] == gen]['hypervolume'].values
#         hv_value = sum(hv_temp) / len(hv_temp)
#         hv_lst.append(hv_value)
#
#     offset_hv = [x - min(hv_lst) for x in hv_lst]
#     norm_hv = [x / (max(offset_hv) + 1) for x in offset_hv]
#     hv_trial_lst.append(norm_hv)
#     # print(norm_hv)
#
#
# hv_gen = np.stack(hv_trial_lst)
# hv_gen_lst = []
# for gen in range(hv_gen.shape[1]):
#     hv_temp =hv_gen[:,gen]
#     hv_gen_lst.append(hv_temp)
#
# # print (hv_gen_lst)
# # print (len(hv_gen_lst))
# fig_verify = plt.figure(figsize=(6, 2))
# mean_hv = np.array([np.mean(a) for a in hv_gen_lst])
# std_hv = np.array([np.std(a) for a in hv_gen_lst])
# x_ref = range(1, n_generations + 1)
# plt.plot(x_ref, mean_hv, color='red', linewidth=1, label = 'Mean')
#
# plt.fill_between(x_ref, mean_hv-std_hv, mean_hv+std_hv,
#     alpha=0.15, facecolor=(1.0, 0.8, 0.8))
#
# plt.plot(x_ref, mean_hv-std_hv, color='black', linewidth= 0.5, linestyle='dashed')
# plt.plot(x_ref, mean_hv+std_hv, color='black', linewidth= 0.5, linestyle='dashed', label = 'Std')
# plt.xticks(x_ref, fontsize=9, rotation=60)
# plt.yticks(fontsize=11)
# plt.ylabel("Normalized hypervolume", fontsize=11)
# plt.xlabel("Generations", fontsize=11)
# plt.legend(loc='lower right', fontsize=12)
# fig_verify.savefig(os.path.join(pic_dir, 'hv_plot_%s_%s.png' % (pop_size, n_generations)), dpi=1500,
#                    bbox_inches='tight')
# fig_verify.savefig(os.path.join(pic_dir, 'hv_plot_%s_%s.eps' % (pop_size, n_generations)), dpi=1500,
#                    bbox_inches='tight')



########################################
spacing_x = 0.5
spacing_y = 500
cycol = cycle('bgrcmk')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



x_max = 11
x_min = 6
y_max = 4000
y_min = 0
x_sp = 0.25
y_sp = 200





############################### Histogram

# Define any condition here
fit_hist_array = np.zeros(int((x_max - x_min)/x_sp)*int((y_max - y_min)/y_sp))
# print (prft_trial_lst[0])
prft_all = pd.concat(prft_trial_lst)
x_bin = []
y_bin = []
print (prft_all)
counter = 0
for idx in range(int((x_max - x_min)/x_sp)) :
    df_fit1 = prft_all.loc[(x_min+idx*x_sp <prft_all[col_a])& (prft_all[col_a]<x_min+(idx+1)*x_sp)]
    for loop in range(int((y_max - y_min)/y_sp)):
        df_fit_temp = df_fit1.loc[(y_min + loop * y_sp < df_fit1[col_b]) & (df_fit1[col_b] < y_min + (loop + 1) * y_sp)]
        # print ("idx", idx)
        # print ("loop", loop)
        # print (df_fit_temp)
        # print (len(df_fit_temp.index))
        fit_hist_array[counter] = fit_hist_array[counter] + len(df_fit_temp.index)
        counter = counter+1
        x_bin.append(x_min+idx*x_sp)
        y_bin.append(y_min + loop * y_sp)


print (fit_hist_array)

# values, edges = np.histogram(fit_hist_array, bins=len(fit_hist_array))
# plt.stairs(values, edges, fill=True)
print (len(fit_hist_array))
print (sum(fit_hist_array))

max_idx = np.argmax(fit_hist_array)
print ("max_idx", max_idx)
print (x_bin[max_idx])
print (y_bin[max_idx])
# plt.hist(fit_hist_array, bins=len(fit_hist_array))
x = np.arange(len(fit_hist_array))
print (x)


fig = matplotlib.figure.Figure(figsize=(3, 3))
agg.FigureCanvasAgg(fig)
cmap = get_cmap(len(prft_trial_lst))
ax = fig.add_subplot(1, 1, 1)
for idx, prft in enumerate(prft_trial_lst):
    # ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1, label="All solutions")
    ax.scatter(prft[col_a], prft[col_b], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1, c=cmap(idx),
               s=20, label="Trial %s" %(idx+1), alpha=0.5)
ax.hlines(np.arange(y_min, y_max, y_sp), 0, 13, lw= 0.5, colors=(0.5, 0.5, 0.5, 0.5), zorder=2)
ax.vlines(np.arange(x_min, x_max, x_sp), 0, 4000, lw= 0.5, colors=(0.5, 0.5, 0.5, 0.5), zorder=2)

rect = matplotlib.patches.Rectangle((x_bin[max_idx],y_bin[max_idx]), x_sp, y_sp, lw=2, facecolor=(0.8, 0.8, 0.1),
                                    alpha = 0.7, edgecolor=  (0.9,0.9,0.1), zorder=1)

# rect = matplotlib.patches.Rectangle((x_bin[max_idx],y_bin[max_idx]), x_sp, y_sp, lw=2, fill=None,
#                                     edgecolor=  (1.0,0.9,0.1), zorder=1)
ax.add_patch(rect)
x_range = np.arange(x_min, x_max, 2*x_sp)
ax.set_xticks(x_range)
ax.set_xticklabels(x_range, rotation=60)
ax.set_yticks(
    np.arange(y_min, y_max, 2*y_sp))
# ax.set_xticklabels(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x), rotation=60)
# if resolution > 0.001:
#     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
#     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
ax.set_xlim(x_min,x_max)
ax.set_ylim(0,y_max)
# ax.set_title("Solutions and pareto front", fontsize=15)
ax.set_xlabel('Validation RMSE', fontsize=12)
ax.set_ylabel('Trainable parameters', fontsize=12)
ax.legend(fontsize=9)
# ax.set_rasterized(True)
fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s.png' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s.eps' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s.pdf' % (pop_size, n_generations)), bbox_inches='tight')

fig_verify = plt.figure(figsize=(6, 4))
plt.bar(x, width=0.8, color= 'r',height=fit_hist_array)
plt.xticks([max_idx], ["fit1: [%s,%s]" %(x_bin[max_idx], x_bin[max_idx]+x_sp) + "\n" + "fit2: [%s,%s]" %(y_bin[max_idx], y_bin[max_idx]+y_sp) ])
plt.ylabel("Counts", fontsize=15)
plt.xlabel("Bins", fontsize=15)
# plt.show()
fig_verify.savefig(os.path.join(pic_dir, 'hist_%s_%s.png' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
fig_verify.savefig(os.path.join(pic_dir, 'hist_%s_%s.eps' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
# fit1_all_lst = prft_all[:,4].tolist()
# fit2_all_lst = prft_all[:,5].tolist()
#
# print ("fit1_all_lst", fit1_all_lst)
#
# for idx in range(len(fit1_hist_array)) :
#     count = sum(x_min+idx*x_sp < x < x_min+(idx+1)*x_sp  for x in fit1_all_lst)
#     fit1_hist_array[idx] = fit1_hist_array[idx] +  count
#
# print (fit1_hist_array)
results_df = pd.read_csv(os.path.join(ea_log_path, "results.csv"))
mlp_solution = [6.79, 94701]
cnn_solution = [6.29, 5722]

hrs_ref = 3600

ga_elm1 = pd.DataFrame([], index=None)
ga_elm1["rmse"] = [7.29, 7.25, 7.37, 7.26, 7.24, 7.19, 7.3, 7.23, 7.2, 7.31]
ga_elm1["params"] = [3310, 3690, 3150, 3060, 3400, 3710, 3360, 3400, 3570, 3460]
ga_elm1["params"] = [3310, 3670, 3150, 3060, 3400, 3690, 3360, 3400, 3550, 3460]
ga_elm1["p2"] = [1870, 1880, 1840, 1750, 1910, 1930, 1770, 1810, 1910, 1790]
ga_elm1["p3"] = [1440, 1790, 1310, 1310, 1490, 1760, 1590, 1590, 1640, 1670]


ga1_time_sec = [68007, 53217, 52735, 66037, 58617, 51156, 61150, 60594, 59981, 51263]
ga1_time_hrs = [x / hrs_ref for x in ga1_time_sec]
ga_elm1["time"] = ga1_time_hrs
ga_elm1["training_time"] = [310, 412, 298, 281, 322, 415, 320, 297, 385, 330]


ga_elm2 = pd.DataFrame([], index=None)
ga_elm2["rmse"] = [7.22, 7.18, 7.22, 7.27, 7.26, 7.14, 7.25, 7.15, 7.22, 7.23]
ga_elm2["params"] = [1770, 1980, 1510, 2270, 1950, 1650, 1790, 1930, 1950, 1790]
# ga_elm2["params"] = [1790, 2000, 1510, 2270, 1950, 1650, 1790, 1930, 1950, 1790]
ga_elm2["p2"] = [1730, 1950, 1500, 1960, 1910, 1570, 1780, 1880, 1900, 1320]
ga_elm2["p3"] = [40, 30, 10, 310, 40, 80, 10, 70, 50, 470]
ga2_time_sec = [32170, 33150, 37140, 34550, 35120, 34830, 36352, 36073, 33075, 35722]
ga2_time_hrs = [x / hrs_ref for x in ga2_time_sec]
ga_elm2["time"] = ga2_time_hrs
ga_elm2["training_time"] = [122, 70, 147, 112, 83, 96, 116, 115, 131, 109]



# ga_elm1 = [7.29, 3310]
# ga_elm2 = [7.22, 1790]

selected_prft = prft_all.loc[(prft_all[col_a] > x_bin[max_idx]) & (prft_all[col_a] < x_bin[max_idx] + x_sp)
                             & (prft_all[col_b] > y_bin[max_idx])
                             & (prft_all[col_b] < y_bin[max_idx] + y_sp)]
print ("selected_prft", selected_prft)
print ("results_df", results_df)
results_df["params"] = selected_prft["fitness_2"].values

fig_results = plt.figure(figsize=(3, 3))

cmap = get_cmap(2)
ax = fig_results.add_subplot(1, 1, 1)

ax.scatter(mlp_solution[0], mlp_solution[1], marker="p",facecolor=(0.9, 0.5, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="MLP")
ax.scatter(cnn_solution[0], cnn_solution[1], marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="CNN")
ax.scatter(ga_elm1["rmse"], ga_elm1["params"], facecolor=(0.7, 1.0, 1.0), edgecolors=(0.2, 0.2, 0.2), zorder=1,
           s=60, label="SOO-ELM(1)", alpha=0.3)
ax.scatter(ga_elm1["rmse"].mean(), ga_elm1["params"].mean(), marker="^",facecolor=(0.0, 0.0, 1.0), edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=70, label="SOO-ELM(1)(avg)", alpha=0.7)
# ax.scatter(ga_elm2[0], ga_elm2[1], marker="s",facecolor=(0.0, 0.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
#            s=60, label="GA+ELM(2)")
ax.scatter(ga_elm2["rmse"], ga_elm2["params"], facecolor=(1,0.8,0.89), edgecolors=(0.2, 0.2, 0.2), zorder=1,
           s=60, label="SOO-ELM(2)", alpha=0.3)
ax.scatter(ga_elm2["rmse"].mean(), ga_elm2["params"].mean(),  marker="s",facecolor=(1.0,0.0,1.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=60, label="SOO-ELM(2)(avg)", alpha=0.7)

ax.scatter(results_df["rmse"], results_df["params"], facecolor=(1.0,1.0,0.0), edgecolors=(0.7, 0.7, 0.0), zorder=1,
           s=60, label="MOO-ELM", alpha=0.7)
ax.scatter(results_df["rmse"].mean(), results_df["params"].mean(),  marker="x",facecolor=(1.0,0.0,0.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=80, label="MOO-ELM(avg)", alpha=1)

#
print ("ga_elm1.mean() rmse", ga_elm1["rmse"].mean())
print ("ga_elm1.mean() params", ga_elm1["params"].mean())
print ("ga_elm1.mean() p2", ga_elm1["p2"].mean())
print ("ga_elm1.mean() p3", ga_elm1["p3"].mean())
print ("ga_elm1.mean() time", ga_elm1["time"].mean())
print ("ga_elm1.mean() training_time", ga_elm1["training_time"].mean())
print ("ga_elm1.std() rmse", ga_elm1["rmse"].std())
print ("ga_elm1.std() params", ga_elm1["params"].std())
print ("ga_elm1.std() p2", ga_elm1["p2"].std())
print ("ga_elm1.std() p3", ga_elm1["p3"].std())
print ("ga_elm1.std() time", ga_elm1["time"].std())
print ("ga_elm1.std() training_time", ga_elm1["training_time"].std())


print ("ga_elm2.mean() rmse", ga_elm2["rmse"].mean())
print ("ga_elm2.mean() params", ga_elm2["params"].mean())
print ("ga_elm2.mean() p2", ga_elm2["p2"].mean())
print ("ga_elm2.mean() p3", ga_elm2["p3"].mean())
print ("ga_elm2.mean() time", ga_elm2["time"].mean())
print ("ga_elm2.mean() training_time", ga_elm2["training_time"].mean())
print ("ga_elm2.std() rmse", ga_elm2["rmse"].std())
print ("ga_elm2.std() params", ga_elm2["params"].std())
print ("ga_elm2.std() p2", ga_elm2["p2"].std())
print ("ga_elm2.std() p3", ga_elm2["p3"].std())
print ("ga_elm2.std() time", ga_elm2["time"].std())
print ("ga_elm2.std() training_time", ga_elm2["training_time"].std())


print ("len(results_df)", len(results_df))
print ("moo results_df.mean() rmse", results_df["rmse"].mean())
print ("results_df.mean() params", results_df["params"].mean())
print ("results_df.mean() p2", selected_prft["p2"].mean())
print ("results_df.mean() p3", selected_prft["p3"].mean())
print ("results_df.mean() f1", selected_prft["fitness_1"].mean())
print ("results_df.mean() f2", selected_prft["fitness_2"].mean())
print ("results_df.mean() train_time", results_df["train_time"].mean())
print ("moo results_df.std() rmse", results_df["rmse"].std())
print ("results_df.std() params", results_df["params"].std())
print ("results_df.std() p2", selected_prft["p2"].std())
print ("results_df.std() p3", selected_prft["p3"].std())
print ("results_df.std() f1", selected_prft["fitness_1"].std())
print ("results_df.std() train_time", results_df["train_time"].std())

moo_time_sec = [22000, 21000 ]
moo_time_hrs = [x / hrs_ref for x in moo_time_sec]
moo_time_avg = np.std(moo_time_hrs)
moo_time_std = np.mean(moo_time_hrs)
print ("moo_time_avg", moo_time_avg)
print ("moo_time_std", moo_time_std)






x_range = np.arange(x_min, 10.5, 2*x_sp)
ax.set_xticks(x_range)
ax.set_xticklabels(x_range, rotation=60)
# ax.set_yticks(
#     np.arange(y_min, 6000, 1000))
# ax.set_xticklabels(np.arange(round(min(data[col_a]), 1)-0.2, round(max(data[col_a]), 1)+0.2, spacing_x), rotation=60)
# if resolution > 0.001:
#     ax.hlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
#     ax.vlines(np.arange(0, 1.4, resolution), 0, 1.4, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
ax.set_xlim(x_min,10.5)
# ax.set_ylim(0,6000)
# ax.set_title("Solutions and pareto front", fontsize=15)


# ax.set_yscale('log')

# exp = lambda x: 100**(x)
# log = lambda x: np.log100(x)

ax.set_yscale('linear')
ax.set_ylim((0, 7000))

y_range = np.arange(0, 7000, 1000)
ax.set_yticks(y_range)

# ax.spines['left'].set_visible(False)
# axMain.yaxis.set_ticks_position('right')
# axMain.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')

divider = make_axes_locatable(ax)
axLin = divider.append_axes("top", size=0.5, pad=0, sharex=ax)
axLin.scatter(mlp_solution[0], mlp_solution[1], marker="p",facecolor=(0.9, 0.5, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="MLP")

axLin.scatter(cnn_solution[0], cnn_solution[1], marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="CNN")
axLin.scatter(ga_elm1["rmse"], ga_elm1["params"], facecolor=(0.7, 1.0, 1.0), edgecolors=(0.2, 0.2, 0.2), zorder=1,
           s=60, label="SOO-ELM(1)", alpha=0.3)
axLin.scatter(ga_elm1["rmse"].mean(), ga_elm1["params"].mean(), marker="^",facecolor=(0.0, 0.0, 1.0), edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=70, label="SOO-ELM(1)(avg)", alpha=0.7)
# ax.scatter(ga_elm2[0], ga_elm2[1], marker="s",facecolor=(0.0, 0.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
#            s=60, label="GA+ELM(2)")
axLin.scatter(ga_elm2["rmse"], ga_elm2["params"], facecolor=(1,0.8,0.89), edgecolors=(0.2, 0.2, 0.2), zorder=1,
           s=60, label="SOO-ELM(2)", alpha=0.3)
axLin.scatter(ga_elm2["rmse"].mean(), ga_elm2["params"].mean(),  marker="s",facecolor=(1.0,0.0,1.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=60, label="SOO-ELM(2)(avg)", alpha=0.7)

axLin.scatter(results_df["rmse"], results_df["params"], facecolor=(1.0,1.0,0.0), edgecolors=(0.7, 0.7, 0.0), zorder=1,
           s=60, label="MOO-ELM", alpha=0.7)
axLin.scatter(results_df["rmse"].mean(), results_df["params"].mean(),  marker="x",facecolor=(1.0,0.0,0.0),  edgecolors=(0.0, 0.0, 0.0), zorder=5,
           s=80, label="MOO-ELM(avg)", alpha=1)

axLin.set_yscale('log')

# axLin.set_yscale('function', functions=(exp, log))
axLin.set_ylim((10000, 180000))
# axLin.plot(np.sin(xdomain), xdomain)
# axLin.spines['right'].set_visible(False)
# axLin.yaxis.set_ticks_position('left')
# plt.setp(axLin.get_xticklabels(), visible=True)


labels = [tick.get_text() for tick in axLin.get_yticklabels()]
print ("labels", labels)
# labels[0]= '10^5'
# # ax2.set_yticklabels(labels)
# print ("labels", labels)
# # axLin.set_yticklabels(labels[1:])
# axLin.set_yticklabels(labels)



axLin.spines['bottom'].set_visible(False)
axLin.xaxis.set_ticks_position('none')
plt.setp(axLin.get_xticklabels(), visible=False)
ax.set_xlabel('Test RMSE', fontsize=12)
# plt.ylabel('Trainable parameters', fontsize=12)
ax.set_ylabel('          Trainable parameters', fontsize=12)
plt.legend(fontsize=8)

fig_results.savefig(os.path.join(pic_dir, 'results_%s_%s.png' % (pop_size, n_generations)), dpi=500, bbox_inches='tight')
fig_results.savefig(os.path.join(pic_dir, 'results_%s_%s.eps' % (pop_size, n_generations)), dpi=500, bbox_inches='tight')
fig_results.savefig(os.path.join(pic_dir, 'results_%s_%s.pdf' % (pop_size, n_generations)), bbox_inches='tight')