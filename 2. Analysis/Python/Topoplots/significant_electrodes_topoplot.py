#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

import mne
import numpy             as np
import os

from matplotlib import pyplot as plt
from matplotlib import rcParams

#%%

rcParams['font.family'] = 'Times New Roman'

#%%

# path to the result of the permutation data
FAST = False
N    = 10

if FAST:
    FOLDER     = "Repetition 1 vs. repetition 8"
    TIME_LIM   = 1.00
    T1, T2     = .700, .850
    F1, F2     = 8, 12
    VMIN, VMAX = -.2, 1
    TITLE      = r"$\alpha$ cluster: contributing electrodes" 
else:
    FOLDER     = "Block 1 vs. block 8 (2)"
    TIME_LIM   = .750
    T1, T2     = .250, .400
    F1, F2     = 4, 8
    VMIN, VMAX = -.5, .1
    TITLE      = r"$\theta$ cluster: contributing electrodes" 

# define the folder where the data and time are stored
PERM_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\{}".format(FOLDER)
TIME_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data"

# define frequency bands (log spaced for setting the y-ticks later on)
FREQS = np.logspace(np.log10(4), 
                    np.log10(30), 
                    15)

#%%

"""
Data loading, averaging and filtering
"""

if FAST:
    print("\n***\nWorking on fast timescale\n***\n")
else:
    print("\n***\nWorking on slow timescale\n***\n")
    
# load the permutation test result array + check dimensions of the data
f_obs       = np.load(os.path.join(PERM_DATA, "f_obs.npy"))
clust       = np.load(os.path.join(PERM_DATA, "clust.npy"), allow_pickle = True)
clust_p_val = np.load(os.path.join(PERM_DATA, "clust_p_val.npy"))

# Create new stats image with only significant clusters
f_obs_plot = np.nan * np.ones_like(f_obs)
for c, p_val in zip(clust, clust_p_val):
    if p_val <= 0.05:
        f_obs_plot[tuple(c)] = f_obs[tuple(c)]

# load the time data, and select everything between 0 and 1s
times = np.load(os.path.join(TIME_DATA, "stimulus_times.npy"))
times = times[np.where((times > .0) & (times <= TIME_LIM))]

# define the time- and frequency window of interest
t = np.where((times > T1) & (times <= T2))
f = np.where((FREQS >= F1) & (FREQS <= F2))

# subset the data to match with our interest
f_obs_plot = f_obs_plot[:,:,t[0]]
f_obs_plot = f_obs_plot[:,f[0],:] 

# take the mean across electrodes, ignoring the NaNs
elec_mean = np.nanmean(f_obs_plot, axis=(1, 2))

# take into account which electrodes are all NaNs, and take this into account
# sort the averages: NaNs are allocated at the end, so we ignore these
if np.isnan(elec_mean).any():
    all_nan = sum(np.isnan(elec_mean))
    print("***\n{0} electrode(s) did not contribute at all.\nIgnored in further analyses\n***\n".format(all_nan))

# get the N electrodes with the largest N values, ignoring full NaNs
elecs_sorted = np.argsort(elec_mean)
selection    = elecs_sorted[-(N + all_nan):-all_nan]

#%%

# create the Montage and Info object to create Evoked array later on
biosemi_montage = mne.channels.make_standard_montage('biosemi64')
fake_info       = mne.create_info(ch_names = biosemi_montage.ch_names, 
                                  sfreq    = 250.,
                                  ch_types = 'eeg')

# create some data, and assign earlier variables to new fake Evoked object
data    = np.zeros((len(biosemi_montage.ch_names), 1))
fake_ev = mne.EvokedArray(data, fake_info)
fake_ev.set_montage(biosemi_montage)

#%%

# get the most active electrodes, and plot them
most_active_n = np.array(biosemi_montage.ch_names)[selection]

print("***\n{} most active electrodes\n***".format(N))

for i, name in enumerate(most_active_n):
    print("{0:d}. {1}".format(i+1, name))
    
#%%

# mask all electrodes except the ones that have a large value
mask            = np.repeat(False, 64)
mask[selection] = True

# create a placeholder for the topoplot (for saving purposes)
fig, ax = plt.subplots(ncols = 1, 
                       figsize = (8, 4), 
                       squeeze = True)

# create the topoplot
# note that the data is all 0, so make the plot darker by decreasing abs(vmin)
# and lighter by increasing abs(vmin)
# note that the non sign. sensors are controlled by keyword sensors, while mask
# controls the N largest sensors
fake_ev.plot_topomap(times       = 0, 
                     colorbar    = False, 
                     vmin        = VMIN,
                     vmax        = VMAX,
                     sensors     = "wo",
                     mask        = mask.reshape(64, 1), 
                     mask_params = dict(marker          = 'o', 
                                        markerfacecolor = 'black', 
                                        markeredgecolor = 'black',
                                        linewidth       = 0, 
                                        markersize      = 15),
                     cmap        = "RdBu_r",
                     outlines    = "skirt", 
                     time_format = " ",
                     size        = 1,
                     res         = 128,
                     title       = " ", 
                     axes        = ax)

# extra plotting parameters
fig.tight_layout()
ax.set_title(TITLE, fontsize = 36, pad = 25.0)
