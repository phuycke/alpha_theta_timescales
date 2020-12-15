#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

import matplotlib.pyplot as plt
import numpy             as np
import os
import pandas            as pd
import seaborn           as sns

from matplotlib import rcParams

#%%

# where to find the data files
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data"

# bool to indicate whether we work stimulus-locked, or response-locked
STIM_LOCKED = True

# which frequency band are we checking
THETA_BAND  = True
ALPHA_BAND  = False
BETA_BAND   = False

# seaborn param
sns.set_style("ticks")
sns.set_context("paper")

#%%

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = 35
rcParams['axes.labelsize']  = 30
rcParams['xtick.labelsize'] = 30
rcParams['ytick.labelsize'] = 30

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

"""
Plotting
All the plotting is done using seaborn, and relies on the data being presented
in long format. That is the reason we did the array reshaping in the previous
step.
"""

x_title, y_title, col_title = None, None, None

# make sure that we are only looking at one frequency band
assert (THETA_BAND + ALPHA_BAND + BETA_BAND) == 1

if STIM_LOCKED:
    if THETA_BAND:
        x_title, y_title, col_title = "Repetition count", "Theta power", "Condition"
    elif ALPHA_BAND:
        x_title, y_title, col_title = "Block number", "Alpha power", "Condition"
    else:
        x_title, y_title, col_title = "Block number", "Beta power", "Condition"

# actual plot
g = sns.catplot(x       = x_title, 
                y       = y_title, 
                col     = col_title,
                data    = df, 
                kind    = "point",
                join    = False,
                capsize = .2,
                color   = "black")

# set the y-axis parameters
if THETA_BAND:
    (g.set_xticklabels(np.arange(1, 9))
      .set(ylim = (-.1, .1)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(r"Stimulus-locked $\theta$ ~ repetition count x condition") 
elif ALPHA_BAND:
    (g.set_xticklabels(np.arange(1, 9))
      .set(ylim = (-.45, -.15)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(r"Stimulus-locked $\alpha$ ~ block number x condition") 
else:
    (g.set_xticklabels(np.arange(1, 9))
      .set(ylim = (-.25, -.15)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(r"Stimulus-locked $\beta$ ~ block number x condition") 

# =============================================================================
# # unused code to save a plot
# g.savefig(os.path.join(ROOT, "Plots", "title.jpg"), 
#           optimize    = True,
#           bbox_inches = "tight")
# =============================================================================

#%%
    

# Novel condition
if STIM_LOCKED:
    if THETA_BAND:
        g = sns.regplot(x           = "Repetition count", 
                        y           = "Theta power", 
                        data        = df, 
                        x_estimator = np.mean, 
                        color       = "black",
                        x_ci        = "ci", 
                        line_kws    = {'lw': 5},
                        ci          = 95,
                        n_boot      = 5000)
        
        plt.subplots_adjust(top=0.9)
        (g.set(ylim = (-.1, .1),
               xlim = (.5, 8.5)))
        g.set_yticks(np.arange(-.1, .11, 0.05))
        g.set_title(r"$\theta$ power ~ fast timescale") 
        g.set_ylabel(r"$\theta$ power")
        g.set_xlabel("Stimulus number")         
    elif ALPHA_BAND:
        # make a subplot with 1 row and 2 columns
        f, ax_list = plt.subplots(1, 2,
                                  sharex  = True, 
                                  sharey  = True,
                                  squeeze = True)
        
        # Novel condition
        g = sns.regplot(x           = "Repetition count", 
                        y           = "Theta power", 
                        data        = df.loc[df["Condition"] == "Novel"], 
                        x_estimator = np.mean, 
                        x_ci        = "ci", 
                        ci          = 95,
                        n_boot      = 5000,
                        line_kws    = {'lw': 5},
                        color       = "darkgrey",
                        ax          = ax_list[0])
        
        # Recurring condition
        g = sns.regplot(x           = "Repetition count", 
                        y           = "Theta power", 
                        data        = df.loc[df["Condition"] == "Recurring"], 
                        x_estimator = np.mean, 
                        x_ci        = "ci", 
                        ci          = 95,
                        n_boot      = 5000,
                        line_kws    = {'lw': 5},
                        color       = "black",
                        ax          = ax_list[1])
        
        
        ax_list[0].set_title(r"Novel condition")   
        ax_list[0].set_ylim([-.1, .1])  
        ax_list[0].set_yticks(np.arange(-.1, .105, .05))   
        ax_list[0].set_xticks(np.arange(1, 9))
        ax_list[0].set_xlim(0.5, 8.5)
        ax_list[0].set_ylabel(r"$\theta$ power")
        
        ax_list[1].set_title(r"Recurring condition")   
        ax_list[1].set_yticks(np.arange(-.1, .105, .05))   
        ax_list[1].set_ylabel(r"$\theta$ power")
        
        f.suptitle(r"$\theta$ power ~ fast timescale")
    else:
        # make a subplot with 1 row and 2 columns
        f, ax_list = plt.subplots(1, 2,
                                  sharex  = True, 
                                  sharey  = True,
                                  squeeze = True)
        
        # Novel condition
        g = sns.regplot(x           = "Block number", 
                        y           = "Beta power", 
                        data        = df.loc[df["Condition"] == "Novel"], 
                        x_estimator = np.mean, 
                        x_ci        = "ci", 
                        ci          = 95,
                        n_boot      = 5000,
                        line_kws    = {'lw': 5},
                        color       = "darkgrey",
                        ax          = ax_list[0])
        
        # Recurring condition
        g = sns.regplot(x           = "Block number", 
                        y           = "Beta power", 
                        data        = df.loc[df["Condition"] == "Recurring"], 
                        x_estimator = np.mean, 
                        x_ci        = "ci", 
                        ci          = 95,
                        n_boot      = 5000,
                        line_kws    = {'lw': 5},
                        color       = "black",
                        ax          = ax_list[1])
        
        
        ax_list[0].set_title(r"Novel condition")   
        ax_list[0].set_ylim([-.25, -.15])  
        ax_list[0].set_yticks(np.arange(-.25, -.145, .05))   
        ax_list[0].set_xticks(np.arange(1, 9))
        ax_list[0].set_xlim(0.5, 8.5)
        ax_list[0].set_ylabel(r"$\beta$ power")
        
        ax_list[1].set_title(r"Repeating condition")   
        ax_list[0].set_ylim([-.25, -.15])  
        ax_list[0].set_yticks(np.arange(-.25, -.145, .025))   
        ax_list[1].set_xlim(0.5, 8.5)
        ax_list[1].set_ylabel(r"$\beta$ power")
        
        f.suptitle(r"$\beta$ power ~ slow timescale")
