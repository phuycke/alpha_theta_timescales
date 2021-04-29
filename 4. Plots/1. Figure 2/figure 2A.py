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

#%%

"""
Plotting
All the plotting is done using seaborn, and relies on the data being presented
in long format. That is the reason we did the array reshaping in the previous
step.
"""

x_title, y_title = "Repetition count", "Reaction time (ms)"
        
# actual plot
g = sns.catplot(x       = x_title, 
                y       = y_title, 
                data    = df, 
                kind    = "point",
                join    = True,
                capsize = .2,
                color   = "black", 
                kwargs  = {"s":15, 'lw': .50})

# plot parameters
plt.ylim(500, 700)
plt.yticks(np.arange(500, 701, 100))
plt.xticks(np.arange(0, 8))
plt.ylabel("RT (in ms)")
plt.xlabel("Stimulus number")
plt.title("Fast timescale")


#%%

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"
fig = plt.gcf()
fig.set_size_inches(3, 2)

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.911, bottom=0.159, left=0.14, right=0.971, 
                    hspace=0.2, wspace=0.2)

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure 2A.tiff"), dpi = 800)
plt.savefig(fname = os.path.join(FIG, "Figure 2A.pdf"), dpi = 800)

plt.close("all")

