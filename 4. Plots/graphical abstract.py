#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

import matplotlib.pyplot as plt
import mne
import numpy             as np
import os
import pandas            as pd
import seaborn           as sns

from scipy      import ndimage
from matplotlib import ticker, rcParams, gridspec

#%%

TEXT_SIZE = 15

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = TEXT_SIZE
rcParams['axes.labelsize']  = TEXT_SIZE
rcParams['xtick.labelsize'] = TEXT_SIZE
rcParams['ytick.labelsize'] = TEXT_SIZE

#%%

# create grid for plots
fig = plt.figure(figsize=(10, 9)) 
gs  = gridspec.GridSpec(2, 21)

ga_a = plt.subplot(gs[0, :7])
ga_b = plt.subplot(gs[0, 11:16])
ga_c = plt.subplot(gs[0, 16:])       

ga_d = plt.subplot(gs[1, :7])
ga_e = plt.subplot(gs[1, 11:16])
ga_f = plt.subplot(gs[1, 16:])    

#%%

"""
Graphical abstract panel A
"""

# Determines which part of the analysis to run + some plotting parameters
STIM_LOCKED = True
COMPUTE_TFR = False
BAND        = [(8, 12, "Alpha")]
TMIN, TMAX  = .65, .9  
VMIN, VMAX  = 0.5, 4.5

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size']   = 8

# these are the subjects that had all 512 epochs recorded and stored safely
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

# load the TFR data
rep1 = mne.time_frequency.read_tfrs(r"C:\Users\pieter\Downloads\repetition 1 (24 subs)-tfr.h5")[0]
rep8 = mne.time_frequency.read_tfrs(r"C:\Users\pieter\Downloads\repetition 8 (24 subs)-tfr.h5")[0]
    
# save rep8 in temp, dB transform
temp = rep8
temp._data = 10 * np.log10(rep8._data)

# save rep1 in temp2, dB transform
temp2 = rep1
temp2._data = 10 * np.log10(rep1._data)

temp._data -= temp2._data
    
# check whether the difference does not equal rep_1 or rep_8
assert np.all(temp._data != rep1._data)
assert not np.sum(temp._data != rep8._data)
    
# colorbar with log scaled labels
def fmt_float(x, pos):
    return r'${0:.2f}$'.format(x)

# define the data
avg_tfr = temp

# get the frequency bands
FMIN, FMAX, FNAME = BAND[0]

# make topoplot
avg_tfr.plot_topomap(tmin     = TMIN,
                     tmax     = TMAX,
                     fmin     = FMIN,
                     fmax     = FMAX,
                     vmin     = VMIN, 
                     vmax     = VMAX,
                     unit     = " ",
                     ch_type  = "eeg", 
                     cmap     = "RdBu_r",
                     outlines = "head", 
                     contours = 10,
                     colorbar = True,
                     cbar_fmt = fmt_float,
                     sensors  = "ko",
                     axes     = ga_a,
                     title    = " ")

# set a title which can be altered
ga_a.set_title(r"$\alpha$ topography", size = TEXT_SIZE)

#%%

"""
Graphical abstract panel B
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
                ax          = ga_b)


# figure parameters (right figure)
ga_b.set_xlim(0.5, 8.5)
ga_b.set_xticks(np.arange(1, 9))
ga_b.set_ylim([-.5, -.1])  
ga_b.set_yticks(np.arange(-.5, -.09, .1))   
ga_b.set_title(r"Repeating condition", size = TEXT_SIZE)   
ga_b.set_xlabel(r"Stimulus number")
ga_b.set_ylabel(r"$\alpha$ power")


#%%

"""
Graphical abstract panel C
"""

# new variables
x_title, y_title = "Block number", "Alpha power"

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
                ax          = ga_c)

# figure parameters (right figure) 
ga_c.set_xlim(0.5, 8.5)
ga_c.set_xticks(np.arange(1, 9))
ga_c.set_ylim([-.5, -.1])  
ga_c.set_yticks(np.arange(-.5, -.09, .1))   
ga_c.set_yticklabels([]) 
ga_c.set_title(r"Repeating condition", size = TEXT_SIZE)   
ga_c.set_xlabel(r"Block number")
ga_c.set_ylabel(" ")


#%%

"""
Graphical abstract panel D
"""

# Determines which part of the analysis to run + some plotting parameters
STIM_LOCKED = True
COMPUTE_TFR = False
BAND        = [(4, 8, "Theta")]
TMIN, TMAX  = 0, .4
VMIN, VMAX  = -.85, .85

# load the TFR data
block1 = mne.time_frequency.read_tfrs(r"C:\Users\pieter\Downloads\block 1 (24 subs)-tfr.h5")[0]
block8 = mne.time_frequency.read_tfrs(r"C:\Users\pieter\Downloads\block 8 (24 subs)-tfr.h5")[0]

# save block8 in temp, dB transform
temp = block8
temp._data = 10 * np.log10(block8._data)

# save block1 in temp2, dB transform
temp2 = block1
temp2._data = 10 * np.log10(block1._data)

temp._data -= temp2._data
    
# check whether the difference does not equal block1 or block8
assert np.all(temp._data != block1._data)
assert not np.sum(temp._data != block8._data)

# colorbar with log scaled labels
def fmt_float(x, pos):
    return r'${0:.2f}$'.format(x)

# define the data
avg_tfr = temp

# get the frequency bands
FMIN, FMAX, FNAME = BAND[0]

# make topoplot
avg_tfr.plot_topomap(tmin     = TMIN,
                     tmax     = TMAX,
                     fmin     = FMIN,
                     fmax     = FMAX,
                     vmin     = VMIN, 
                     vmax     = VMAX,
                     unit     = " ",
                     ch_type  = "eeg", 
                     cmap     = "RdBu_r",
                     outlines = "head", 
                     contours = 10,
                     colorbar = True,
                     cbar_fmt = fmt_float,
                     sensors  = "ko",
                     axes     = ga_d,
                     title    = " ")

# set a title which can be altered
ga_d.set_title(r"$\theta$ topography", size = TEXT_SIZE)

#%%

"""
Graphical abstract panel E
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
x_title, y_title = "Repetition count", "Theta power"

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
                ax          = ga_e)

# figure parameters (right figure)
ga_e.set_title(r"Repeating condition", size = TEXT_SIZE)   
ga_e.set_ylim([-.1, .1])  
ga_e.set_yticks(np.arange(-.1, .11, .1))   
ga_e.set_xticks(np.arange(1, 9))
ga_e.set_xlim(0.5, 8.5)
ga_e.set_xlabel(r"Stimulus number")
ga_e.set_ylabel(r"$\theta$ power")

#%%

"""
Graphical abstract panel F
"""

# new variables
x_title, y_title = "Block number", "Theta power"

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
                ax          = ga_f)

# figure parameters (right figure)
ga_f.set_title(r"Repeating condition", size = TEXT_SIZE)   
ga_f.set_ylim([-.1, .1])  
ga_f.set_yticks(np.arange(-.1, .11, .1))   
ga_f.set_yticklabels([])   
ga_f.set_xticks(np.arange(1, 9))
ga_f.set_xlim(0.5, 8.5)
ga_f.set_xlabel(r"Block number")
ga_f.set_ylabel("")

#%%

"""
Save figure
"""

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.90, bottom=0.11, left=0.11, right=0.9, hspace=0.35, 
                    wspace=1.0)

# dB labels
plt.text(-1.9, 3.93, "dB", size = TEXT_SIZE)    
plt.text(-1.9, 0.90, "dB", size = TEXT_SIZE)    

# topographies label
plt.text(-28, 4.49, "Topographies", size = TEXT_SIZE+5)    

# fast slow timescale
plt.text(25, 4.49, "Fast timescale", size = TEXT_SIZE+5)    
plt.text(53.5, 4.49, "Slow timescale", size = TEXT_SIZE+5)    

# alpha theta band
plt.text(-45, 2.75, r"$\alpha$ band", size = TEXT_SIZE + 5, rotation = 90)    
plt.text(-45, -0.24, r"$\theta$ band", size = TEXT_SIZE + 5, rotation = 90)    

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Graphical abstract.tiff"), dpi = 300)
plt.savefig(fname = os.path.join(FIG, "Graphical abstract.pdf"), dpi = 300)

plt.close("all")


