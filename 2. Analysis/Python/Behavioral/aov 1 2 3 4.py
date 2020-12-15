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

# which frequency band are we checking
FAST = False

# seaborn param
sns.set_style("ticks")
sns.set_context("paper", font_scale = 2.5)
rcParams['font.family'] = 'Times New Roman'

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

if FAST:
    x_title, y_title, col_title = "Repetition count", "Reaction time (ms)", None
else:
    x_title, y_title, col_title = "Block number", "Reaction time (ms)", "Condition"

        
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
if FAST:
    (g.set_xticklabels(np.arange(1, 9))
      .set(ylim = (475, 600)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("RT (in ms) ~ block number x condition") 
else:
    (g.set_xticklabels(np.arange(1, 9))
      .set(ylim = (475, 600)))
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("RT (in ms) ~ repetition count") 

# =============================================================================
# # unused code to save a plot
# g.savefig(os.path.join(ROOT, "Plots", "title.jpg"), 
#           optimize    = True,
#           bbox_inches = "tight")
# =============================================================================

#%%
    

# Novel condition
if FAST:
    g = sns.regplot(x           = x_title, 
                    y           = y_title, 
                    data        = df, 
                    x_estimator = np.mean, 
                    color       = "black",
                    x_ci        = "ci", 
                    line_kws    = {'lw': 5},
                    ci          = 95,
                    n_boot      = 5000)
    
    plt.subplots_adjust(top=0.9)
    (g.set(ylim = (450, 600),
           xlim = (.5, 8.5)))
    g.set_yticks(np.arange(450, 601, 50))
    g.set_title(r"RT (ms) ~ fast timescale") 
    g.set_ylabel(r"RT (in ms)")
    g.set_xlabel("Repetition count")
else:
    # make a subplot with 1 row and 2 columns
    f, ax_list = plt.subplots(1, 2,
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
                    line_kws    = {'lw': 5},
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
                    line_kws    = {'lw': 5},
                    color       = "black",
                    ax          = ax_list[1])
    
    
    ax_list[0].set_title(r"Novel condition")   
    ax_list[0].set_ylim([450, 600])  
    ax_list[0].set_yticks(np.arange(450, 601, 50))   
    ax_list[0].set_xticks(np.arange(1, 9))
    ax_list[0].set_xlim(0.5, 8.5)
    ax_list[0].set_ylabel(r"RT (ms)")
    
    ax_list[1].set_title(r"Recurring condition")   
    ax_list[1].set_ylim([450, 600])  
    ax_list[1].set_yticks(np.arange(450, 601, 50))   
    ax_list[1].set_xticks(np.arange(1, 9))
    ax_list[1].set_xlim(0.5, 8.5)
    ax_list[1].set_ylabel(r"RT (ms)")
    
    f.suptitle(r"RT ~ slow timescale") 
    
#%%

# create a space for two plots next to each other
f, axes = plt.subplots(2, 1)

# create a function to do the conversion from absolute count to relative count
def convert_to_relative(dataframe = pd.DataFrame):
    
    # melt the dataframe by grouping it according to the needed dimensions
    prop_df = (dataframe[x]
               .groupby(dataframe[hue])
               .value_counts(normalize=False)
               .rename(y)
               .reset_index())
    
    # get the absolute count for each block number and each accuracy label
    # calculate the relative frequency of (in)correct and overwrite the original
    for i in range(1, 9):
        sub_df    = prop_df.loc[prop_df[x] == i]
        sub_df[y] = (sub_df[y] / sub_df[y].sum()) * 100
        for i in range(len(sub_df)):
            indx                 = sub_df.index[i]
            prop_df.iloc[indx,2] = sub_df.iloc[i, 2]
    
    return prop_df

# define some of the plotting parameters
x, y, hue = "Block number", "Percentage", "Accuracy"
hue_order = ["Correct", "Wrong"]

# subset the data per condition: not repeating
indices   = np.where(df["Condition"] == "Novel")[0]
subsetted = df.iloc[indices]
    
# call the function on the subsetted data
temp = convert_to_relative(dataframe = subsetted)

# make the actual barplot in seaborn
sns.barplot(x         = x, 
            y         = y, 
            hue       = hue, 
            hue_order = ["Wrong", "Correct"],
            data      = temp,
            palette   = sns.color_palette(["black", "w"]),
            edgecolor = "black",
            ax        = axes[0],
            linewidth = 1.5)

# subset the data per condition: repeating
indices   = np.where(df["Condition"] == "Recurring")[0]
subsetted = df.iloc[indices]
    
# call the function on the subsetted data
temp = convert_to_relative(dataframe = subsetted)

# make the actual barplot in seaborn
sns.barplot(x         = x, 
            y         = y, 
            hue       = hue,
            hue_order = ["Wrong", "Correct"], 
            data      = temp,
            palette   = sns.color_palette(["black", "w"]),
            edgecolor = "black",
            ax        = axes[1],
            linewidth = 1.5)

del temp

# ---- extra plotting parameters ----
# add a legend on each axis
for ax in axes:
    ax.legend(bbox_to_anchor = (1.05, 1), 
              loc            = "upper center",
              borderaxespad  = 0.,
              ncol           = 1, 
              prop           = {'size': 14})

# correct the tick labels on the y-axes
for ax in axes:
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels(np.arange(0, 101, 10))


for ax in axes:
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels(np.arange(0, 101, 10))

titles = [None, "Block number"]
for i, ax in enumerate(axes):
    ax.set_xlabel(titles[i])

# add an overarching title
f.suptitle('Accuracy ~ slow timescale', fontsize = 16)

# add specific titles for each plot
titles = ["Novel", "Recurring"]
for ax in axes:
    ax.set_title("Condition: {}".format(titles[f.axes.index(ax)]), 
                 fontsize = 16)

#%%

# subset the data for a number of repetitions <= 8
indices   = np.where(df["Repetitions_overall"] <= 8)[0]
subsetted = df.iloc[indices]

# create a space for the plot
f, ax = plt.subplots(1, 1)

# create a function to do the conversion from absolute count to relative count
def convert_to_relative(dataframe = pd.DataFrame):
    
    # melt the dataframe by grouping it according to the needed dimensions
    prop_df = (dataframe[x]
               .groupby(dataframe[hue])
               .value_counts(normalize=False)
               .rename(y)
               .reset_index())
    
    # get the absolute count for each block number and each accuracy label
    # calculate the relative frequency of (in)correct and overwrite the original
    for i in range(1, 9):
        sub_df    = prop_df.loc[prop_df[x] == i]
        sub_df[y] = (sub_df[y] / sub_df[y].sum()) * 100
        for i in range(len(sub_df)):
            indx                 = sub_df.index[i]
            prop_df.iloc[indx,2] = sub_df.iloc[i, 2]
    
    return prop_df

# define some of the plotting parameters
x, y, hue = "Repetition count", "Percentage", "Accuracy"

# call the function on the subsetted data
temp = convert_to_relative(dataframe = subsetted)
temp = temp.iloc[np.where(temp["Accuracy"] == "Correct")[0]]

# make the actual barplot in seaborn
sns.barplot(x         = x, 
            y         = y,
            data      = temp,
            color     = "white",
            edgecolor = "black",
            linewidth = 1.5)
del temp

# ---- extra plotting parameters ----

# correct the tick labels on the axes
ax.set_xticklabels(np.arange(1, 9))
ax.set_yticks(np.arange(0, 101, 10))
ax.set_yticklabels(np.arange(0, 101, 10))
ax.set_ylabel("Correct (in %)")

# add an overarching title
f.suptitle('Accuracy ~ fast timescale', fontsize = 16)

