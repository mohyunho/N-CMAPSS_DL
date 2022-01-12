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
from scipy.optimize import curve_fit
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
current_dir = os.path.dirname(os.path.abspath(__file__))
corr_log_path = os.path.join(current_dir, 'corr_log.csv')
corr_log_df = pd.read_csv(corr_log_path, dtype=np.float64)

pic_dir = os.path.join(current_dir, 'Figures')

print(corr_log_df)



def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def func_lin(x, a, b):
    return x*a+b

def func_sqr(x, a, b, c):
    return x*x*a + x*b + c

xdata = corr_log_df['train_params'].values
print (type(xdata[0]))
ydata = corr_log_df['train_time'].values
xdata = xdata.astype(np.float64)
print (type(xdata[0]))
ydata = ydata.astype(np.float64)
print (xdata)
print (ydata)

fig = plt.figure(figsize=(9, 2))

plt.plot(xdata, ydata, 'go', label='Data')

popt, pcov = curve_fit(func_lin, xdata, ydata)
# print (pcov)
print (popt)
# plt.plot(xdata, func_lin(xdata, *popt), 'b-', label='Linear: a=%5.2e, b=%5.2e' % tuple(popt))
popt, pcov = curve_fit(func_sqr, xdata, ydata)
# print (pcov)
print (popt)
plt.plot(xdata, func_sqr(xdata, *popt), 'r-', label='Best-fit curve, quadratic: a=%5.2e, b=%5.2e, c=%5.2e' % tuple(popt))
plt.legend(fontsize=11)
plt.ylabel("Training time", fontsize=13)
plt.xlabel("Trainable parameters", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.ylim([0,500])

fig.savefig(os.path.join(pic_dir, 'corr_plot.png' ), dpi=1500 ,bbox_inches='tight')
fig.savefig(os.path.join(pic_dir, 'corr_plot.eps' ), dpi=1500 ,bbox_inches='tight')