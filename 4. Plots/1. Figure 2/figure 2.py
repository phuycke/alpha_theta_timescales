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

# seaborn param
sns.set_style("ticks")
sns.set_context("paper")

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = TEXT_SIZE + 3
rcParams['axes.labelsize']  = TEXT_SIZE
rcParams['xtick.labelsize'] = TEXT_SIZE
rcParams['ytick.labelsize'] = TEXT_SIZE

#%%

# create grid for plots
fig = plt.figure(figsize=(10, 9)) 
gs  = gridspec.GridSpec(2, 9)

# fast timescale 
fig_2a = plt.subplot(gs[0, :4])
fig_2b = plt.subplot(gs[1, :4])

# slow timescale
fig_2c = plt.subplot(gs[0, 5:])
fig_2d = plt.subplot(gs[1, 5:])

#%%
# where to find the data files
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data"

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

# subset the data for stimulus repetitions
df = df[df['Repetitions_overall'] <= 8]

# change type of relevant variables
df = df.astype({'Repetition count':np.int32, 
                'Reaction time (ms)':np.float32})

# variables plotted
x_title, y_title = "Repetition count", "Reaction time (ms)"
        
# use pointplot to allow for ax use
sns.pointplot(x       = x_title, 
              y       = y_title, 
              data    = df, 
              color   = "black", 
              join    = True, 
              capsize = .2, 
              ax      = fig_2a)

# plot parameters
fig_2a.set_ylim(500, 700)
fig_2a.set_yticks(np.arange(500, 701, 100))
fig_2a.set_xticks(np.arange(0, 8))
fig_2a.set_ylabel("RT (in ms)")
fig_2a.set_xlabel("Stimulus number")
fig_2a.set_title("Fast timescale")

#%%

"""
Figure 2B
"""

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

# subset the data for stimulus repetitions
df = df[df['Repetitions_overall'] <= 8]

# add a column with error rate
dictionary       = {"Correct": 1, "Wrong": 0}
df["Accuracy"]   = df["Accuracy"].map(dictionary)
df["Error rate"] = np.abs((df["Accuracy"] - 1) * 100)

# change type of relevant variables
df = df.astype({'Repetition count':np.int32, 
                'Error rate':np.float32})

x_title, y_title = "Repetition count", "Error rate"
        
# actual plot
sns.pointplot(x       = x_title, 
              y       = y_title, 
              data    = df, 
              color   = "black", 
              join    = True, 
              capsize = .2, 
              ax      = fig_2b)

# plot parameters
fig_2b.set_ylim(0, 50)
fig_2b.set_yticks(np.arange(0, 51, 25))
fig_2b.set_xticks(np.arange(0, 8))
fig_2b.set_ylabel("Error (in %)")
fig_2b.set_xlabel("Stimulus number")
fig_2b.set_title("Fast timescale")

#%%

"""
Figure 2C
"""

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

# add a column with error rate
dictionary       = {"Correct": 1, "Wrong": 0}
df["Accuracy"]   = df["Accuracy"].map(dictionary)
df["Condition"]  = df["Condition"].map({"Novel": "Novel", "Recurring": "Repeating"})
df["Error rate"] = np.abs((df["Accuracy"] - 1) * 100)

# change type of relevant variables
df = df.astype({'Block number':np.int32, 
                'Error rate':np.float32})

x_title, y_title, hue = "Block number", "Reaction time (ms)", "Condition"
        
# actual plot
sns.pointplot(x       = x_title, 
              y       = y_title, 
              hue     = hue,
              data    = df, 
              palette = sns.color_palette(["silver", "black"]),
              join    = True,
              capsize = .2,
              color   = "black", 
              ax      = fig_2c)

# plot parameters
fig_2c.set_ylim(450, 625)
fig_2c.set_yticks(np.arange(500, 601, 100))
fig_2c.set_xticks(np.arange(0, 8))
fig_2c.set_ylabel("RT (in ms)")
fig_2c.set_xlabel("Block number")
fig_2c.set_title("Slow timescale")

# remove title from legend
fig_2c.legend(fontsize = "large", ncol = 2)

#%%

"""
Figure 2D
"""

# read the data
df = pd.read_csv(os.path.join(ROOT, "theta_alpha_beta_behavioural.csv"))

# change the column names to their appropriate label
df.columns = ['Reaction time (ms)', 'RT_log', 'Accuracy', 'Accuracy_int', 
              'Error_int', 'Theta power', 'Alpha power', 'Beta power', 
              'Subject nr', 'Repetitions_overall', 'Repetition count', 
              'Block_overall', 'Block number', 'Condition', 'Trial_overall', 
              'Trial_block', 'Response', 'Stimulus_ID']

# add a column with error rate
dictionary       = {"Correct": 1, "Wrong": 0}
df["Accuracy"]   = df["Accuracy"].map(dictionary)
df["Condition"]  = df["Condition"].map({"Novel": "Novel", "Recurring": "Repeating"})
df["Error rate"] = np.abs((df["Accuracy"] - 1) * 100)

# change type of relevant variables
df = df.astype({'Block number':np.int32, 
                'Error rate':np.float32})

x_title, y_title, hue = "Block number", "Error rate", "Condition"
        
# actual plot
sns.pointplot(x       = x_title, 
              y       = y_title, 
              hue     = hue,
              data    = df, 
              palette = sns.color_palette(["silver", "black"]),
              join    = True,
              capsize = .2,
              color   = "black", 
              ax      = fig_2d)


# plot parameters
fig_2d.set_ylim(0, 10)
fig_2d.set_yticks(np.arange(0, 11, 5))
fig_2d.set_xticks(np.arange(0, 8))
fig_2d.set_ylabel("Error (in %)")
fig_2d.set_xlabel("Block number")
fig_2d.set_title("Slow timescale")

# remove title from legend
fig_2d.legend(fontsize = "large", ncol = 2)

#%%

"""
Save figure
"""

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.958, bottom=0.079, left=0.083, right=0.986,
                    hspace=0.272, wspace=0.0)

# letters indicating the panels
plt.text(-10.25, 21.75, "A", size = TEXT_SIZE+5)    
plt.text(-10.25, 9.1, "B", size = TEXT_SIZE+5)    
plt.text(-.35, 21.75, "C", size = TEXT_SIZE+5)    
plt.text(-.35, 9.1, "D", size = TEXT_SIZE+5)    

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure 2.tiff"), dpi = 800)
plt.savefig(fname = os.path.join(FIG, "Figure 2.pdf"), dpi = 800)

plt.close("all")

