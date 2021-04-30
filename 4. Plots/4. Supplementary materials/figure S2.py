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
import seaborn           as sns

from scipy      import ndimage
from matplotlib import rcParams, gridspec, ticker

#%%

TEXT_SIZE = 12

# seaborn param
sns.set_style("ticks")
sns.set_context("paper")

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = TEXT_SIZE
rcParams['axes.labelsize']  = TEXT_SIZE
rcParams['xtick.labelsize'] = TEXT_SIZE
rcParams['ytick.labelsize'] = TEXT_SIZE

#%%

# create grid for plots
fig = plt.figure(figsize=(10, 6)) 
gs  = gridspec.GridSpec(2, 9)

fig_s2a = plt.subplot(gs[0, :4])
fig_s2b = plt.subplot(gs[0, 5:])

fig_s2c = plt.subplot(gs[1, :4])
fig_s2d = plt.subplot(gs[1, 5:])

#%%

"""
Figure S2 A
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
fig_s2a.pcolormesh(time_mesh, 
                   yval_mesh, 
                   gauss,
                   cmap    = "RdBu_r",
                   shading = "gouraud")

# draw a contour around larger values
# we draw the contour around values that are percentile 97.5 or larger
fig_s2a.contour(time_mesh, 
               yval_mesh, 
               gauss, 
               levels     = [np.percentile(gauss, 97.5)], 
               colors     = "black", 
               linewidths = 3,
               linestyles = "solid")

# set the y-axis parameters, note that the y-axis needs to be converted to 
# log, and that a ticker needs to be called to set the y-axis ticks
fig_s2a.set_yscale('log')
fig_s2a.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
fig_s2a.yaxis.set_minor_formatter(ticker.NullFormatter())
fig_s2a.yaxis.set_minor_locator(ticker.NullLocator())

# once the ticks are set, we assign the values of FREQS to the ticks
tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 15).round().astype('int'))]
fig_s2a.set_yticks(tick_vals)

# determine the y-ticks
ticks_str = []
for t in tick_vals:
    if round(t) in [4, 8, 13, 19, 30]:
        ticks_str.append("{0:.2f}".format(t))
    else:
        ticks_str.append(" ")
fig_s2a.set_yticklabels(ticks_str)

# set the x-axis parameters: every 100 ms a label is placed
fig_s2a.set_xticks(np.arange(0, 1.1, .25))
fig_s2a.set_xticklabels([str(int(label)) for label in np.arange(0, 1001, 250)])

# set the general title, and the titles of the x-axis and the y-axis
fig_s2a.set_xlabel('Time after stimulus (ms)')
fig_s2a.set_ylabel('Frequency (Hz)')
fig_s2a.set_title("Stimulus 1 vs. 8: permutation test TFR\nAlpha on the fast timescale (p = 0.001)")

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
im = fig_s2a.imshow(f_obs_plot_mean,
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
cbar = fig.colorbar(im, ax = fig_s2a)

# set some colorbar parameters, such as the title, ticks and tick labels
cbar.ax.set_title("F-statistic", 
                  fontdict = {"fontsize": TEXT_SIZE})
cbar.ax.get_yaxis().set_ticks(np.arange(0, np.round(np.max(f_obs_plot_mean), 1) + 0.05, 4))
cbar.ax.tick_params(labelsize = TEXT_SIZE - 3)

# big fix: make sure that the 0 is shown on the x-axis of the final plot 
fig_s2a.set_xbound(0, 1)

#%%

"""
Figure S2 B
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


# load in data of one subject to access channel information
ROOT  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\sub-02\ses-01\eeg\Python'
TF    = os.path.join(ROOT, 'time-frequency')
ep    = mne.read_epochs(os.path.join(TF, 'sub-02_stimulus_tf_epo.fif'))   

# load the cluster data, and keep only the significant clusters
clust       = np.load(os.path.join(PERM_DATA, "clust.npy"), allow_pickle = True)
clust_p_val = np.load(os.path.join(PERM_DATA, "clust_p_val.npy"))
f_obs_plot  = np.zeros_like(f_obs)
for c, p_val in zip(clust, clust_p_val):
    if p_val <= 0.05:
        f_obs_plot[tuple(c)] = f_obs[tuple(c)]

# read channels, select time window of interest and create all True mask
# note that the results are the same for frequencies 8 - 14 Hz and 8 - 30 Hz
channels = ep.info["ch_names"][:64]
twoi     = f_obs_plot[:,5:10,717:870]
excl     = np.full(64, True)

# for each electrode, check whether at least 1 datapoint belongs to sign. cluster
for i in range(len(twoi)):
   if np.all(twoi[i,:,:] == 0):
       print("Removed: {0:s}".format(channels[i]))
       excl[i] = False
       
# keep only electrodes for which we have significant cluster points
print("{0:.2f}% of the electrodes ignored".format(100-(sum(excl)/len(excl)*100)))
f_obs = f_obs[excl,:,:]
        
# 64: electrodes, 15: frequencies, 1024: time points
# we average over electrodes to retain the frequency and time information
f_obs_mean = np.mean(f_obs, axis = 0)

# apply a gaussian filter to the data, with SD = 1 for both axes
gauss = ndimage.filters.gaussian_filter(f_obs_mean, 
                                        [1, 1], 
                                        mode = 'constant')

# create a pseudocolor plot
fig_s2b.pcolormesh(time_mesh, 
                   yval_mesh, 
                   gauss,
                   cmap    = "RdBu_r",
                   shading = "gouraud")

# draw a contour around larger values
# we draw the contour around values that are percentile 97.5 or larger
fig_s2b.contour(time_mesh, 
                yval_mesh, 
                gauss, 
                levels     = [np.percentile(gauss, 97.5)], 
                colors     = "black", 
                linewidths = 3,
                linestyles = "solid")

# set the y-axis parameters, note that the y-axis needs to be converted to 
# log, and that a ticker needs to be called to set the y-axis ticks
fig_s2b.set_yscale('log')
fig_s2b.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
fig_s2b.yaxis.set_minor_formatter(ticker.NullFormatter())
fig_s2b.yaxis.set_minor_locator(ticker.NullLocator())

# once the ticks are set, we assign the values of FREQS to the ticks
tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 15).round().astype('int'))]
fig_s2b.set_yticks(tick_vals)

# determine the y-ticks
ticks_str = []
for t in tick_vals:
    if round(t) in [4, 8, 13, 19, 30]:
        ticks_str.append("{0:.2f}".format(t))
    else:
        ticks_str.append(" ")
fig_s2b.set_yticklabels(ticks_str)

# set the x-axis parameters: every 100 ms a label is placed
fig_s2b.set_xticks(np.arange(0, 1.1, .25))
fig_s2b.set_xticklabels([str(int(label)) for label in np.arange(0, 1001, 250)])

# set the general title, and the titles of the x-axis and the y-axis
fig_s2b.set_xlabel('Time after stimulus (ms)')
fig_s2b.set_ylabel('Frequency (Hz)')
fig_s2b.set_title("Stimulus 1 vs. 8: permutation test TFR\nOnly significant electrodes")

# load the cluster data, and keep only the significant clusters
clust       = np.load(os.path.join(PERM_DATA, "clust.npy"), allow_pickle = True)
clust_p_val = np.load(os.path.join(PERM_DATA, "clust_p_val.npy"))
f_obs_plot  = np.zeros_like(f_obs)
for c, p_val in zip(clust, clust_p_val):
    if p_val <= 0.05:
        f_obs_plot[tuple(c)] = f_obs[tuple(c)]
        
# take the average (excluding NaNs) of the significant data
f_obs_plot_mean = np.nanmean(f_obs_plot[excl,:,:], axis = 0)

# create a 2D raster of the significant data (no plot) to use for the colorbar
im = fig_s2b.imshow(f_obs_plot_mean,
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
cbar = fig.colorbar(im, ax = fig_s2b)

# set some colorbar parameters, such as the title, ticks and tick labels
cbar.ax.set_title("F-statistic", 
                  fontdict = {"fontsize": TEXT_SIZE})
cbar.ax.get_yaxis().set_ticks(np.arange(0, np.round(np.max(f_obs_plot_mean), 1) + 0.05, 4))
cbar.ax.tick_params(labelsize = TEXT_SIZE - 3)

# big fix: make sure that the 0 is shown on the x-axis of the final plot 
fig_s2b.set_xbound(0, 1)

#%%

"""
Figure S2 C
"""

# set the path to the data, and load a fitting Epoch object
PERM_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Block 1 vs. block 8 (2)"
TIME_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data"

# define frequency bands (log spaced for setting the y-ticks later on)
FREQS = np.logspace(np.log10(4), 
                    np.log10(30), 
                    15)

# load the time data, and select everything between 0 and 750 ms
times     = np.load(os.path.join(TIME_DATA, "stimulus_times.npy"))
times     = times[np.where((times > 0) & (times <= .75))]

# the the difference between x[0] and x[1] for each value in times, and divide 
# by 2 if len(times) is larger than 750 ms, else fix this at 0.0005
time_diff = np.diff(times) / 2. if len(times) > 1 else [0.0005]

# compute the limits of the time window (x-axis)
    # start:  first value of time (a bit larger than 0) - 0.00048828
    # middle: all values except the last + 0.00048828
    # final:  last value of time (.75) + 0.00048828
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
assert f_obs.shape == (64, 15, 768)

# 64: electrodes, 15: frequencies, 768: time points
# we average over electrodes to retain the frequency and time information
f_obs_mean = np.mean(f_obs, axis = 0)

# apply a gaussian filter to the data, with SD = 1 for both axes
gauss = ndimage.filters.gaussian_filter(f_obs_mean, 
                                        [1, 1], 
                                        mode = 'constant')

# create a pseudocolor plot
fig_s2c.pcolormesh(time_mesh, 
                  yval_mesh, 
                  gauss,
                  cmap    = "RdBu_r",
                  shading = "gouraud")

# draw a contour around larger values
# we draw the contour around values that are percentile 97.5 or larger
fig_s2c.contour(time_mesh, 
               yval_mesh, 
               gauss, 
               levels     = [np.percentile(gauss, 97.5)], 
               colors     = "black", 
               linewidths = 3,
               linestyles = "solid")

# set the y-axis parameters, note that the y-axis needs to be converted to 
# log, and that a ticker needs to be called to set the y-axis ticks
fig_s2c.set_yscale('log')
fig_s2c.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
fig_s2c.yaxis.set_minor_formatter(ticker.NullFormatter())
fig_s2c.yaxis.set_minor_locator(ticker.NullLocator())

# once the ticks are set, we assign the values of FREQS to the ticks
tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 15).round().astype('int'))]
fig_s2c.set_yticks(tick_vals)

# determine the y-ticks
ticks_str = []
for t in tick_vals:
    if round(t) in [4, 8, 13, 19, 30]:
        ticks_str.append("{0:.2f}".format(t))
    else:
        ticks_str.append(" ")
fig_s2c.set_yticklabels(ticks_str)

# set the x-axis parameters: every 50 ms a label is placed
fig_s2c.set_xticks(np.arange(0, .751, .25))
fig_s2c.set_xticklabels([str(int(label)) for label in np.arange(0, 751, 250)])

# set the general title, and the titles of the x-axis and the y-axis
fig_s2c.set_xlabel('Time after stimulus (ms)')
fig_s2c.set_ylabel('Frequency (Hz)')
fig_s2c.set_title("Block 1 vs. 8: permutation test TFR\nTheta on slow timescale (p = 0.04)")

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
im = fig_s2c.imshow(f_obs_plot_mean,
                    extent        = [times[0], times[-1], 
                                    FREQS[0], FREQS[-1]],
                    aspect        = "auto", 
                    origin        = "lower", 
                    interpolation = "hanning",
                    cmap          = "RdBu_r")

# get the colorbar of the above 2D raster, and paste it on the existing TFR plot
cbar = fig.colorbar(im, ax = fig_s2c)

# set some colorbar parameters, such as the title, ticks and tick labels
cbar.ax.set_title("F-statistic", 
                  fontdict = {"fontsize": TEXT_SIZE})
cbar.ax.get_yaxis().set_ticks(np.arange(0, np.round(np.max(f_obs_plot_mean), 1) + 0.05, 2))
cbar.ax.tick_params(labelsize = TEXT_SIZE - 3)

# big fix: make sure that the 0 is shown on the x-axis of the final plot 
fig_s2c.set_xbound(0, .75)

#%%

"""
Figure S2 D
"""

# set the path to the data, and load a fitting Epoch object
PERM_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Block 1 vs. block 8 (2)"
TIME_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data"

# define frequency bands (log spaced for setting the y-ticks later on)
FREQS = np.logspace(np.log10(4), 
                    np.log10(30), 
                    15)

# load the time data, and select everything between 0 and 750 ms
times     = np.load(os.path.join(TIME_DATA, "stimulus_times.npy"))
times     = times[np.where((times > 0) & (times <= .75))]

# the the difference between x[0] and x[1] for each value in times, and divide 
# by 2 if len(times) is larger than 750 ms, else fix this at 0.0005
time_diff = np.diff(times) / 2. if len(times) > 1 else [0.0005]

# compute the limits of the time window (x-axis)
    # start:  first value of time (a bit larger than 0) - 0.00048828
    # middle: all values except the last + 0.00048828
    # final:  last value of time (.75) + 0.00048828
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
assert f_obs.shape == (64, 15, 768)
     
# load in data of one subject to access channel information
ROOT  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\sub-02\ses-01\eeg\Python'
TF    = os.path.join(ROOT, 'time-frequency')
ep    = mne.read_epochs(os.path.join(TF, 'sub-02_stimulus_tf_epo.fif'))   

# load the cluster data, and keep only the significant clusters
clust       = np.load(os.path.join(PERM_DATA, "clust.npy"), allow_pickle = True)
clust_p_val = np.load(os.path.join(PERM_DATA, "clust_p_val.npy"))
f_obs_plot  = np.zeros_like(f_obs)
for c, p_val in zip(clust, clust_p_val):
    if p_val <= 0.05:
        f_obs_plot[tuple(c)] = f_obs[tuple(c)]

# read channels, select time window of interest and create all True mask
channels = ep.info["ch_names"][:64]
twoi     = f_obs_plot[:,:6,256:410]
excl     = np.full(64, True)

# for each electrode, check whether at least 1 datapoint belongs to sign. cluster
for i in range(len(twoi)):
   if np.all(twoi[i,:,:] == 0):
       print("Removed: {0:s}".format(channels[i]))
       excl[i] = False
       
# keep only electrodes for which we have significant cluster points
print("{0:.2f}% of the electrodes ignored".format(100-(sum(excl)/len(excl)*100)))
f_obs = f_obs[excl,:,:]
        
# 64: electrodes, 15: frequencies, 768: time points
# we average over electrodes to retain the frequency and time information
f_obs_mean = np.mean(f_obs, axis = 0)

# apply a gaussian filter to the data, with SD = 1 for both axes
gauss = ndimage.filters.gaussian_filter(f_obs_mean, 
                                        [1, 1], 
                                        mode = 'constant')

# create a pseudocolor plot
fig_s2d.pcolormesh(time_mesh, 
                   yval_mesh, 
                   gauss,
                   cmap    = "RdBu_r",
                   shading = "gouraud")

# draw a contour around larger values
# we draw the contour around values that are percentile 97.5 or larger
fig_s2d.contour(time_mesh, 
                yval_mesh, 
                gauss, 
                levels     = [np.percentile(gauss, 97.5)], 
                colors     = "black", 
                linewidths = 3,
                linestyles = "solid")
    
# set the y-axis parameters, note that the y-axis needs to be converted to 
# log, and that a ticker needs to be called to set the y-axis ticks
fig_s2d.set_yscale('log')
fig_s2d.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
fig_s2d.yaxis.set_minor_formatter(ticker.NullFormatter())
fig_s2d.yaxis.set_minor_locator(ticker.NullLocator())

# once the ticks are set, we assign the values of FREQS to the ticks
tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 15).round().astype('int'))]
fig_s2d.set_yticks(tick_vals)

# determine the y-ticks
ticks_str = []
for t in tick_vals:
    if round(t) in [4, 8, 13, 19, 30]:
        ticks_str.append("{0:.2f}".format(t))
    else:
        ticks_str.append(" ")
fig_s2d.set_yticklabels(ticks_str)


# set the x-axis parameters: every 50 ms a label is placed
fig_s2d.set_xticks(np.arange(0, .751, .25))
fig_s2d.set_xticklabels([str(int(label)) for label in np.arange(0, 751, 250)])

# set the general title, and the titles of the x-axis and the y-axis
fig_s2d.set_xlabel('Time after stimulus (ms)')
fig_s2d.set_ylabel('Frequency (Hz)')
fig_s2d.set_title("Block 1 vs. 8: permutation test TFR\nOnly significant electrodes")

# take the average (excluding NaNs) of the significant data
# Note: we only take into account the significant electrodes
f_obs_plot_mean = np.nanmean(f_obs_plot[excl,:,:], axis = 0)

# create a 2D raster of the significant data (no plot) to use for the colorbar
im = fig_s2d.imshow(f_obs_plot_mean,
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
cbar = fig.colorbar(im, ax = fig_s2d)

# set some colorbar parameters, such as the title, ticks and tick labels
cbar.ax.set_title("F-statistic", 
                  fontdict = {"fontsize": TEXT_SIZE})
cbar.ax.get_yaxis().set_ticks(np.arange(0, np.round(np.max(f_obs_plot_mean), 1) + 0.05, 2))
cbar.ax.tick_params(labelsize = TEXT_SIZE - 3)

# big fix: make sure that the 0 is shown on the x-axis of the final plot 
fig_s2d.set_xbound(0, .75)

#%%

"""
Save figure
"""

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"

# play around until the figure is satisfactory (difficult with high DPI)
plt.subplots_adjust(top=0.92, bottom=0.11, left=0.1, right=1.0, hspace=0.65,
                    wspace=0.0)

# letters indicating the panels
plt.text(-1.35, 1000, "A", size = TEXT_SIZE+5)    
plt.text(-.18, 1000, "B", size = TEXT_SIZE+5)    
plt.text(-1.35, 37, "C", size = TEXT_SIZE+5)    
plt.text(-.18, 37, "D", size = TEXT_SIZE+5)    

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure S2.tiff"), dpi = 300)
plt.savefig(fname = os.path.join(FIG, "Figure S2.pdf"), dpi = 300)

plt.close("all")


