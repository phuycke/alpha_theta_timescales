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

# add a column with error rate
dictionary       = {"Correct": 1, "Wrong": 0}
df["Accuracy"]   = df["Accuracy"].map(dictionary)
df["Error rate"] = np.abs((df["Accuracy"] - 1) * 100)

#%%

"""
Plotting
All the plotting is done using seaborn, and relies on the data being presented
in long format. That is the reason we did the array reshaping in the previous
step.
"""

x_title, y_title, hue = "Block number", "Error rate", "Condition"
        
# actual plot
g = sns.catplot(x       = x_title, 
                y       = y_title, 
                hue     = hue,
                data    = df, 
                palette = sns.color_palette(["silver", "black"]),
                kind    = "point",
                join    = True,
                capsize = .2,
                color   = "black")

# plot parameters
plt.ylim(0, 10)
plt.yticks(np.arange(0, 11, 5))
(g.set_xticklabels(np.arange(1, 9))
  .set_xlabels("Block number")
  .set_ylabels("Error (in %)")
  .set_yticklabels(["{0:d}".format(i) for i in np.arange(0, 11, 5)]))

