#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

import copy
import matplotlib.pyplot as plt
import numpy             as np
import os
import pandas            as pd
import seaborn           as sns

#%%

# define what the x variable is, and what the y-variable is
X = "Repetition count"
Y = "Alpha power"

#%%

def custom_regplot(*args, 
                   line_kws    = None, 
                   marker      = None, 
                   scatter_kws = None,
                   **kwargs):
    
    # this is the class that `sns.regplot` uses
    plotter = sns.regression._RegressionPlotter(*args, **kwargs)

    # this is essentially the code from `sns.regplot`
    ax = kwargs.get("ax", None)
    if ax is None:
        ax = plt.gca()

    scatter_kws = {} if scatter_kws is None else copy.copy(scatter_kws)
    scatter_kws["marker"] = marker
    line_kws = {} if line_kws is None else copy.copy(line_kws)

    # uncomment following line to create a plot each time (time consuming)
    # plotter.plot(ax, scatter_kws, line_kws)

    # unfortunately the regression results aren't stored, so we rerun
    grid, yhat, err_bands = plotter.fit_regression(plt.gca())

    # also unfortunately, this doesn't return the parameters, so we infer them
    slope     = (yhat[-1] - yhat[0]) / (grid[-1] - grid[0])
    intercept = yhat[0] - slope * grid[0]
    return intercept, slope

#%%

# where to find the data files
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data"

#%%

"""
load the data: the data was originally stored in a NumPy array. Then it was
converted to a Pandas DataFrame (DF), from where it was converted to a .csv file
We will read the csv file to a Pandas DF from where it will be incorporated in
seaborn plotting code
"""

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

#%%

# keep part of the data
selected = df[['Theta power', 'Alpha power', 'Repetition count', 
               'Block number', 'Condition', 'Subject nr']]
selected = selected[(selected[X] > 1) & (selected[X] < 8)]
                    
# create a place holder for the data
arr = np.zeros((len(np.unique(selected["Subject nr"])), 2))

#%%

for i, sub in enumerate(np.unique(selected["Subject nr"])):
    for j, cond in enumerate(np.unique(selected["Condition"])):
        
        # where are we 
        print("Working on subject {0:d} {1:s} condition".format(int(sub), cond.lower()))
        
        # subset relevant data
        mask = ((selected["Subject nr"] == sub) & (selected["Condition"] == cond))
        d    = selected[mask]
        
        # get the slope and the intercept
        intercept, slope = custom_regplot(x = X, 
                                          y = Y, 
                                          data = d, 
                                          x_estimator = np.mean, 
                                          x_ci = "ci", 
                                          ci = 95,
                                          n_boot = 5000,
                                          line_kws = {'lw': 5},
                                          color = "darkgrey")

        # save only the slope
        arr[i,j] = slope

# close all figures
plt.close()

#%%

"""
Compute the t-test
"""

import scipy.stats as stats

m1, sd1 = np.round(np.mean(arr[:,0]), 1), np.round(np.std(arr[:,0]), 1)
m2, sd2 = np.round(np.mean(arr[:,1]), 1), np.round(np.std(arr[:,1]), 1)

print(m1, sd1)
print(m2, sd2)

t_stat, p_val = stats.ttest_rel(arr[:,0], arr[:,1])
print("T-statistic: {}\np-value: {}".format(np.round(t_stat, 3),
                                            np.round(p_val, 3)))