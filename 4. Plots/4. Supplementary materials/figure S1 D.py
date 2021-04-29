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

# seaborn param
sns.set_style("ticks")
sns.set_context("paper")

#%%

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = 6
rcParams['axes.labelsize']  = 5
rcParams['xtick.labelsize'] = 5
rcParams['ytick.labelsize'] = 5

#%%

"""
load the data: the data was originally stored in a NumPy array. Then it was
converted to a Pandas DataFrame (DF), from where it was converted to a .csv file
We will read the csv file to a Pandas DF from where it will be incorporated in
seaborn plotting code
"""

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_1elec_behavioural.csv"))

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

x_title, y_title = "Block number", "Theta power"

#%%
    
# make a subplot with 1 row and 2 columns
fig, ax_list = plt.subplots(1, 2,
                            sharex  = True, 
                            sharey  = True,
                            squeeze = True)

# Novel condition
g = sns.regplot(x           = x_title, 
                y           = y_title, 
                data        = df.loc[df["Condition"] == "Novel"], 
                x_estimator = np.mean, 
                x_ci        = "ci", 
                ci          = 95,
                n_boot      = 5000,
                scatter_kws = {"s":15}, 
                line_kws    = {'lw': .75},
                color       = "darkgrey",
                ax          = ax_list[0])

# Recurring condition
g = sns.regplot(x           = x_title, 
                y           = y_title, 
                data        = df.loc[df["Condition"] == "Recurring"], 
                x_estimator = np.mean, 
                x_ci        = "ci", 
                ci          = 95,
                n_boot      = 5000,
                scatter_kws = {"s":15}, 
                line_kws    = {'lw': .75},
                color       = "black",
                ax          = ax_list[1])

# figure parameters (left figure)
ax_list[0].set_title(r"Novel condition")   
ax_list[0].set_ylim([-.15, .15])  
ax_list[0].set_yticks(np.arange(-.15, .16, .05))   
ax_list[0].set_xticks(np.arange(1, 9))
ax_list[0].set_xlim(0.5, 8.5)
ax_list[0].set_xlabel(r"Block number")
ax_list[0].set_ylabel(r"$\theta$ power")

# figure parameters (right figure)
ax_list[1].set_title(r"Repeating condition")   
ax_list[1].set_xlabel(r"Block number")
ax_list[1].set_ylabel(r"$\theta$ power")

# general title
fig.suptitle(r"$\theta$ power ~ slow timescale (Pz)", fontsize = 8) 

#%%

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"
fig.set_size_inches(3, 2)

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.185, right=0.95, hspace=0.075,
                    wspace=0.2)

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure S1 D.tiff"), dpi = 800)
plt.savefig(fname = os.path.join(FIG, "Figure S1 D.pdf"), dpi = 800)

plt.close("all")

