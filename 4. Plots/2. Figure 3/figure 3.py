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

TEXT_SIZE = 20

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = TEXT_SIZE
rcParams['axes.labelsize']  = TEXT_SIZE
rcParams['xtick.labelsize'] = TEXT_SIZE
rcParams['ytick.labelsize'] = TEXT_SIZE

#%%

# create grid for plots
fig = plt.figure(figsize=(6, 4)) 
gs  = gridspec.GridSpec(2, 4)

# TFR plot
fig_3a   = plt.subplot(gs[0, :3])

# topoplot
fig_3b   = plt.subplot(gs[0, -1:])

# alpha on the fast timescale
fig_3c_l = plt.subplot(gs[1, 0])  # novel condition
fig_3c_r = plt.subplot(gs[1, 1])  # repeating condition

# alpha on the slow timescale
fig_3d_l = plt.subplot(gs[1, 2])  # novel condition
fig_3d_r = plt.subplot(gs[1, 3])  # repeating condition

#%%

"""
Figure 3A
"""

# path to the result of the permutation data
PERM_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Repetition 1 vs. repetition 8"
TIME_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data"

# define frequency bands (log spaced for setting the y-ticks later on)
FREQS = np.logspace(np.log10(4), 
                    np.log10(30), 
                    15)

# load the time data, and select everything between 0 and 1s
times     = np.load(os.path.join(TIME_DATA, "stimulus_times.npy"))
times     = times[np.where((times > 0) & (times <= 1))]

# the the difference between x[0] and x[1] for each value in times, and divide 
# by 2 if len(times) is larger than 1s, else fix this at 0.0005
time_diff = np.diff(times) / 2. if len(times) > 1 else [0.0005]

# compute the limits of the time window (x-axis)
    # start:  first value of time (a bit larger than 0) - 0.00048828
    # middle: all values except the last + 0.00048828
    # final:  last value of time (1) + 0.00048828
time_lims = np.concatenate([[times[0] - time_diff[0]], times[:-1] +
                            time_diff, [times[-1] + time_diff[-1]]])

# get the values that should be on the y-axis
yvals = FREQS

# compute the ratio: x[1] = x[0] * ratio (holds for all values)
ratio = yvals[1:] / yvals[:-1]

# compute the limits of the frequencies (y-axis) 
    # start:  first value of yvals (4) / 1.15479362
    # middle: the values of yvals
    # last:   the last value of yvals (30) * 1.15479362
log_yvals = np.concatenate([[yvals[0] / ratio[0]], yvals,
                            [yvals[-1] * ratio[0]]])

# get the limits of the y-axis
    # note that yvals_lims is in this case equal to yvals since yvals is 
    # log-spaced. This would not be true if linspace was used to get frequencies
yval_lims = np.sqrt(log_yvals[:-2] * log_yvals[2:])
time_lims = time_lims[:-1]

# create a meshgrid
    # time_mesh: row values are the same, column values differ (time)
    # yval_mesh: row values differ (freqs), column values are the same
time_mesh, yval_mesh = np.meshgrid(time_lims, yval_lims)

# load the permutation test result array + check dimensions of the data
f_obs = np.load(os.path.join(PERM_DATA, "f_obs.npy"))
assert f_obs.shape == (64, 15, 1024)

# 64: electrodes, 15: frequencies, 1024: time points
# we average over electrodes to retain the frequency and time information
f_obs_mean = np.mean(f_obs, axis = 0)

# apply a gaussian filter to the data, with SD = 1 for both axes
gauss = ndimage.filters.gaussian_filter(f_obs_mean, 
                                        [1, 1], 
                                        mode = 'constant')

# create a pseudocolor plot
fig_3a.pcolormesh(time_mesh, 
              yval_mesh, 
              gauss,
              cmap    = "RdBu_r",
              shading = "gouraud")

# draw a contour around larger values
# we draw the contour around values that are percentile 97.5 or larger
fig_3a.contour(time_mesh, 
           yval_mesh, 
           gauss, 
           levels     = [np.percentile(gauss, 97.5)], 
           colors     = "black", 
           linewidths = 3,
           linestyles = "solid")

# set the y-axis parameters, note that the y-axis needs to be converted to 
# log, and that a ticker needs to be called to set the y-axis ticks
fig_3a.set_yscale('log')
fig_3a.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
fig_3a.yaxis.set_minor_formatter(ticker.NullFormatter())
fig_3a.yaxis.set_minor_locator(ticker.NullLocator())

# once the ticks are set, we assign the values of FREQS to the ticks
tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 15).round().astype('int'))]
fig_3a.set_yticks(tick_vals)

# determine the y-ticks
ticks_str = []
for t in tick_vals:
    if round(t) in [4, 8, 13, 19, 30]:
        ticks_str.append("{0:.2f}".format(t))
    else:
        ticks_str.append(" ")
fig_3a.set_yticklabels(ticks_str)

# set the x-axis parameters: every 100 ms a label is placed
fig_3a.set_xticks(np.arange(0, 1.1, .25))
fig_3a.set_xticklabels([str(int(label)) for label in np.arange(0, 1001, 250)])

# set the general title, and the titles of the x-axis and the y-axis
fig_3a.set_xlabel('Time after stimulus (ms)')
fig_3a.set_ylabel('Frequency (Hz)')
fig_3a.set_title("Stimulus 1 vs. 8: permutation test TFR\nAlpha on the fast timescale (p = 0.001)")

# load the cluster data, and keep only the significant clusters
clust       = np.load(os.path.join(PERM_DATA, "clust.npy"), allow_pickle = True)
clust_p_val = np.load(os.path.join(PERM_DATA, "clust_p_val.npy"))
f_obs_plot  = np.zeros_like(f_obs)
for c, p_val in zip(clust, clust_p_val):
    if p_val <= 0.05:
        f_obs_plot[tuple(c)] = f_obs[tuple(c)]
        
# take the average (excluding NaNs) of the significant data
f_obs_plot_mean = np.nanmean(f_obs_plot, axis = 0)

# create a 2D raster of the significant data (no plot) to use for the colorbar
im = fig_3a.imshow(f_obs_plot_mean,
               extent        = [times[0], times[-1], 
                                FREQS[0], FREQS[-1]],
               aspect        = "auto", 
               origin        = "lower", 
               interpolation = "hanning",
               cmap          = "RdBu_r")

# get the colorbar of the above 2D raster, and paste it on the existing TFR plot
# note that this data is used to create the colorbar, and not the filtered data
# since the values become smaller due to the filtering process. The plot reflects
# the actual data, filtering is only done for visual appeal
cbar = fig.colorbar(im, ax = fig_3a)

# set some colorbar parameters, such as the title, ticks and tick labels
cbar.ax.set_title("F-statistic", 
                  fontdict = {"fontsize": TEXT_SIZE})
cbar.ax.get_yaxis().set_ticks(np.arange(0, np.round(np.max(f_obs_plot_mean), 1) + 0.05, 4))
cbar.ax.tick_params(labelsize = 12)

# big fix: make sure that the 0 is shown on the x-axis of the final plot 
fig_3a.set_xbound(0, 1)

#%%

"""
Figure 3B
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
                     axes     = fig_3b,
                     title    = " ")

# set a title which can be altered
fig_3b.set_title(r"$\alpha$ topography", size = TEXT_SIZE)

#%%

"""
Figure 3C
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
                ax          = fig_3c_l)

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
                ax          = fig_3c_r)

# figure parameters (left figure)
fig_3c_l.set_title(r"Novel condition", size = TEXT_SIZE)   
fig_3c_l.set_ylim([-.5, -.1])  
fig_3c_l.set_yticks(np.arange(-.5, -.09, .1))   
fig_3c_l.set_xticks(np.arange(1, 9))
fig_3c_l.set_xlim(0.5, 8.5)
fig_3c_l.set_xlabel(r"Stimulus number")
fig_3c_l.set_ylabel(r"$\alpha$ power")

# figure parameters (right figure)
fig_3c_r.set_ylim([-.5, -.1])  
fig_3c_r.set_yticks(np.arange(-.5, -.09, .1))   
fig_3c_r.set_yticklabels([])   
fig_3c_r.set_xlim(0.5, 8.5)
fig_3c_r.set_xticks(np.arange(1, 9))
fig_3c_r.set_title(r"Repeating condition", size = TEXT_SIZE)   
fig_3c_r.set_xlabel(r"Stimulus number")
fig_3c_r.set_ylabel(r"$\alpha$ power")

#%%

"""
Figure 3D
"""

# new variables
x_title, y_title = "Block number", "Alpha power"

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
                ax          = fig_3d_l)

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
                ax          = fig_3d_r)

# figure parameters (left figure)
fig_3d_l.set_title(r"Novel condition", size = TEXT_SIZE)   
fig_3d_l.set_ylim([-.5, -.1])  
fig_3d_l.set_yticks(np.arange(-.5, -.09, .1))   
fig_3d_l.set_xticks(np.arange(1, 9))
fig_3d_l.set_xlim(0.5, 8.5)
fig_3d_l.set_xlabel(r"Block number")
fig_3d_l.set_ylabel(r"$\alpha$ power")

# figure parameters (right figure)
fig_3d_r.set_ylim([-.5, -.1])  
fig_3d_r.set_yticks(np.arange(-.5, -.09, .1))   
fig_3d_r.set_yticklabels([])   
fig_3d_r.set_xlim(0.5, 8.5)
fig_3d_r.set_xticks(np.arange(1, 9))
fig_3d_r.set_title(r"Repeating condition", size = TEXT_SIZE)   
fig_3d_r.set_xlabel(r"Block number")
fig_3d_r.set_ylabel(r"$\alpha$ power")

#%%

"""
Save figure
"""

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.924, bottom=0.087, left=0.062, right=0.967, 
                    hspace=0.323, wspace=0.305)

# TODO: manual additions
plt.text(-1.5, 4.6, "dB")     # Figure 3B colorbar scale
# alpha ~ fast timescale title
# alpha ~ slow timescale title

# TODO: fix plot for 300 DPI

# =============================================================================
# # save as tiff and pdf
# plt.savefig(fname = os.path.join(FIG, "Figure 3.tiff"), dpi = 300)
# plt.savefig(fname = os.path.join(FIG, "Figure 3.pdf"), dpi = 300)
# 
# plt.close("all")
# 
# =============================================================================
