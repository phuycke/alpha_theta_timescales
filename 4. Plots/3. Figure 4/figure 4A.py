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

from scipy      import ndimage
from matplotlib import ticker
from matplotlib import rcParams

#%%

rcParams['font.family']     = 'Times New Roman'
rcParams['axes.titlesize']  = 12
rcParams['axes.labelsize']  = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12

#%%

# set the path to the data, and load a fitting Epoch object
PERM_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Block 1 vs. block 8 (2)"
TIME_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data"

# define frequency bands (log spaced for setting the y-ticks later on)
FREQS = np.logspace(np.log10(4), 
                    np.log10(30), 
                    15)

#%%

# create a plotting handle
fig, ax = plt.subplots(1,1)

#%%

"""
Compute bounds between time samples and construct a time-yvalue bounds grid
"""

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

#%%

"""
Data loading, averaging and filtering
"""

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

#%%

"""
Plotting the filtered values of the permutation test + circling the relevant
values based on percentile
"""

# create a pseudocolor plot
ax.pcolormesh(time_mesh, 
              yval_mesh, 
              gauss,
              cmap    = "RdBu_r",
              shading = "gouraud")

# draw a contour around larger values
# we draw the contour around values that are percentile 97.5 or larger
ax.contour(time_mesh, 
           yval_mesh, 
           gauss, 
           levels     = [np.percentile(gauss, 97.5)], 
           colors     = "black", 
           linewidths = 3,
           linestyles = "solid")

#%%

"""
General plotting parameters: y-axis, x_axis, and titles
"""

# set the y-axis parameters, note that the y-axis needs to be converted to 
# log, and that a ticker needs to be called to set the y-axis ticks
ax.set_yscale('log')
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_minor_locator(ticker.NullLocator())

# once the ticks are set, we assign the values of FREQS to the ticks
tick_vals = yvals[np.unique(np.linspace(0, len(yvals) - 1, 15).round().astype('int'))]
ax.set_yticks(tick_vals)

# determine the y-ticks
ticks_str = []
for t in tick_vals:
    if round(t) in [4, 8, 13, 19, 30]:
        ticks_str.append("{0:.2f}".format(t))
    else:
        ticks_str.append(" ")
ax.set_yticklabels(ticks_str)


# set the x-axis parameters: every 50 ms a label is placed
ax.set_xticks(np.arange(0, .751, .25))
ax.set_xticklabels([str(int(label)) for label in np.arange(0, 751, 250)])

# set the general title, and the titles of the x-axis and the y-axis
plt.xlabel('Time after stimulus (ms)')
plt.ylabel('Frequency (Hz)')
plt.title("Block 1 vs. 8: permutation test TFR\nTheta on slow timescale (p = 0.04)")

#%%

"""
Get the colorbar for this plot, note that this involves read
"""

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
im = ax.imshow(f_obs_plot_mean,
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
cbar = fig.colorbar(im, ax = ax)

# set some colorbar parameters, such as the title, ticks and tick labels
cbar.ax.set_title("F-statistic", 
                  fontdict = {"fontsize": 25})
cbar.ax.get_yaxis().set_ticks(np.arange(0, np.round(np.max(f_obs_plot_mean), 1) + 0.05, 2))
cbar.ax.tick_params(labelsize = 25)

#%%

# big fix: make sure that the 0 is shown on the x-axis of the final plot 
ax.set_xbound(0, .75)

#%%

# define the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"
fig.set_size_inches(6, 4)

# play around until the figure is satisfactory
plt.subplots_adjust(top=0.89, bottom=0.15, left=0.13, right=0.992, hspace=0.1,
                    wspace=0.1)

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure 3A.tiff"), dpi = 300)
plt.savefig(fname = os.path.join(FIG, "Figure 3A.pdf"), dpi = 300)

plt.close("all")

