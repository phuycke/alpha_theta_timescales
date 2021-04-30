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

from matplotlib import rcParams, gridspec

#%%

TEXT_SIZE = 15

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = TEXT_SIZE
rcParams['axes.labelsize']  = TEXT_SIZE
rcParams['xtick.labelsize'] = TEXT_SIZE
rcParams['ytick.labelsize'] = TEXT_SIZE

#%%

# create grid for plots
fig = plt.figure(figsize=(10, 6)) 
gs  = gridspec.GridSpec(2, 13)

# panel A
fig_s1a_l   = plt.subplot(gs[0, 0:3])
fig_s1a_r   = plt.subplot(gs[0, 3:6])

# panel B
fig_s1b_l = plt.subplot(gs[0, 7:10])      
fig_s1b_r = plt.subplot(gs[0, 10:13])

# panel C
fig_s1c_l = plt.subplot(gs[1, 0:3])       
fig_s1c_r = plt.subplot(gs[1, 3:6])       
 
# panel D
fig_s1d_l = plt.subplot(gs[1, 7:10])      
fig_s1d_r = plt.subplot(gs[1, 10:13])     

#%%

"""
Figure S1 A
"""

# where to find the data files
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data"

# seaborn param
sns.set_style("ticks")
sns.set_context("paper")

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']
x_title, y_title = "Repetition count", "Alpha power"

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
                ax          = fig_s1a_l)

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
                ax          = fig_s1a_r)

# figure parameters (left figure)
fig_s1a_l.set_title(r"Novel condition", size = TEXT_SIZE)   
fig_s1a_l.set_ylim([-.5, -.1])  
fig_s1a_l.set_yticks(np.arange(-.5, -.09, .1))   
fig_s1a_l.set_xticks(np.arange(1, 9))
fig_s1a_l.set_xlim(0.5, 8.5)
fig_s1a_l.set_xlabel(r"Stimulus number")
fig_s1a_l.set_ylabel(r"$\alpha$ power")

# figure parameters (right figure)
fig_s1a_r.set_xlim(0.5, 8.5)
fig_s1a_r.set_xticks(np.arange(1, 9))
fig_s1a_r.set_ylim([-.5, -.1])  
fig_s1a_r.set_yticks(np.arange(-.5, -.09, .1))   
fig_s1a_r.set_yticklabels([])   
fig_s1a_r.set_title(r"Repeating condition", size = TEXT_SIZE)   
fig_s1a_r.set_xlabel(r"Stimulus number")
fig_s1a_r.set_ylabel("")

#%%

"""
Figure S1 B
"""

# where to find the data files
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data"


# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_1elec_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

x_title, y_title = "Repetition count", "Alpha power"

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
                ax          = fig_s1b_l)

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
                ax          = fig_s1b_r)

# figure parameters (left figure)
fig_s1b_l.set_title(r"Novel condition", size = TEXT_SIZE)   
fig_s1b_l.set_ylim([-.6, -.1])  
fig_s1b_l.set_yticks(np.arange(-.6, -.09, .1))   
fig_s1b_l.set_xticks(np.arange(1, 9))
fig_s1b_l.set_xlim(0.5, 8.5)
fig_s1b_l.set_xlabel(r"Stimulus number")
fig_s1b_l.set_ylabel(r"$\alpha$ power")

# figure parameters (right figure)
fig_s1b_r.set_title(r"Repeating condition", size = TEXT_SIZE)   
fig_s1b_r.set_ylim([-.6, -.1])  
fig_s1b_r.set_yticks(np.arange(-.6, -.09, .1))   
fig_s1b_r.set_yticklabels([]) 
fig_s1b_r.set_xticks(np.arange(1, 9))
fig_s1b_r.set_xlim(0.5, 8.5)
fig_s1b_r.set_xlabel(r"Stimulus number")
fig_s1b_r.set_ylabel("")

#%%

"""
Figure S1 C
"""

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

# new variables
x_title, y_title = "Block number", "Theta power"

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
                ax          = fig_s1c_l)

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
                ax          = fig_s1c_r)

# figure parameters (left figure)
fig_s1c_l.set_title(r"Novel condition", size = TEXT_SIZE)   
fig_s1c_l.set_ylim([-.1, .1])  
fig_s1c_l.set_yticks(np.arange(-.1, .11, .1))   
fig_s1c_l.set_xticks(np.arange(1, 9))
fig_s1c_l.set_xlim(0.5, 8.5)
fig_s1c_l.set_xlabel(r"Block number")
fig_s1c_l.set_ylabel(r"$\theta$ power")

# figure parameters (right figure)
fig_s1c_r.set_title(r"Repeating condition", size = TEXT_SIZE)   
fig_s1c_r.set_ylim([-.1, .1])  
fig_s1c_r.set_yticks(np.arange(-.1, .11, .1))   
fig_s1c_r.set_yticklabels([])   
fig_s1c_r.set_xticks(np.arange(1, 9))
fig_s1c_r.set_xlim(0.5, 8.5)
fig_s1c_r.set_xlabel(r"Block number")
fig_s1c_r.set_ylabel("")

#%%

"""
Figure S1 D
"""

# where to find the data files
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data"

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_1elec_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']


x_title, y_title = "Block number", "Theta power"
    
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
                ax          = fig_s1d_l)

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
                ax          = fig_s1d_r)

# figure parameters (left figure)
fig_s1d_l.set_title(r"Novel condition", size = TEXT_SIZE)   
fig_s1d_l.set_ylim([-.15, .15])  
fig_s1d_l.set_yticks(np.arange(-.15, .16, .1))   
fig_s1d_l.set_xticks(np.arange(1, 9))
fig_s1d_l.set_xlim(0.5, 8.5)
fig_s1d_l.set_xlabel(r"Block number")
fig_s1d_l.set_ylabel(r"$\theta$ power")

# figure parameters (right figure)
fig_s1d_r.set_title(r"Repeating condition", size = TEXT_SIZE)   
fig_s1d_r.set_ylim([-.15, .15])  
fig_s1d_r.set_yticks(np.arange(-.15, .16, .1))   
fig_s1d_r.set_yticklabels([])   
fig_s1d_r.set_xticks(np.arange(1, 9))
fig_s1d_r.set_xlim(0.5, 8.5)
fig_s1d_r.set_xlabel(r"Block number")
fig_s1d_r.set_ylabel("")

#%%

"""
Save figure
"""

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.1, right=0.95, hspace=0.75,
                    wspace=1.0)

# letters indicating the panels
plt.text(-35, .72, "A", size = TEXT_SIZE+5)    
plt.text(-12.5, .72, "B", size = TEXT_SIZE+5)    
plt.text(-35, .2, "C", size = TEXT_SIZE+5)    
plt.text(-12.5, .2, "D", size = TEXT_SIZE+5)    

# general titles
plt.text(-28, .75, r"$\alpha$ power ~ fast timescale", size = TEXT_SIZE)
plt.text(-7, .75, r"$\alpha$ power ~ fast timescale (Pz)", size = TEXT_SIZE)
plt.text(-28, .22, r"$\theta$ power ~ slow timescale", size = TEXT_SIZE)
plt.text(-7, .22, r"$\theta$ power ~ slow timescale (Pz)", size = TEXT_SIZE)

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure S1.tiff"), dpi = 800)
plt.savefig(fname = os.path.join(FIG, "Figure S1.pdf"), dpi = 800)

plt.close("all")


