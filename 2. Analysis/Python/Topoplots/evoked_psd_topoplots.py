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

from matplotlib import rcParams

#%%

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size']   = 30

#%%

"""
Globals
Variables that will be used throughout the script. Marked by all capitals in 
line with convention in the C language.
"""

# Determines which part of the analysis to run
STIM_LOCKED = True
FAST        = True

if FAST:
    TIMESCALE     = "Fast"
    TITLE         = ["Repetitions: 1", "Repetitions 8", 
                     "Repetitions 1 - Repetitions 8"]
    COND          = "Overall_repetitions"
    TIME, WINDOW  = .775, .15   
    BAND          = [(8, 12, "Alpha")]
    TMIN, TMAX    = .65, .9  
    DIMS_TO_MATCH = ["Subject_number", "Block_number_specific", "Stimulus_ID", 
                     "Response", "Block_ID"]
    VLIM          = (1e-09, 3.5e-09)
    TFR           = "stim-locked repetition 8 - repetition 1 (non phase)-tfr.h5"
else:
    TIMESCALE     = "Slow"
    TITLE         = ["Block: 1", "Block 8", "Block 1 - block 8"]
    COND          = "Block_number_specific"
    TIME, WINDOW  = .325, .15   
    BAND          = [(4, 8, "Theta")]
    TMIN, TMAX    = .25, .4
    DIMS_TO_MATCH = ["Subject_number", "In_block_repetitions", "Stimulus_ID", 
                     "Response", "Block_ID"]
    VLIM          = (1e-9, 3e-9)
    TFR           = "stim-locked block 8 - block 1 (non phase)-tfr.h5"

# the root folder, but split up for readability
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3"
DATA = os.path.join(ROOT, "figures", "TF", "Group level", "data")
FIG  = os.path.join(ROOT, "figures", "TF", "Group level", "plots")

#%%

"""
Helper functions
"""

# colorbar in scientific notation
def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

#%%

# these are the subjects that we use for further analysis
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

#%%

"""
Store the data for our 24 subjects in a list, then concatenate all the data into
one long Epoch object for further manipulation
"""

epochs_list = []

# get a list of all possible subject numbers, and loop over them
for SUB_ID in full_epochs:     

    # path depends on the subject ID, hence no global def
    PATH = ROOT + r"\{}\ses-01\eeg\Python".format(SUB_ID)
    TF   = os.path.join(PATH, "time-frequency")
            
    # read the relevant epoch data and store in epochs_list
    if STIM_LOCKED:
        epoch = mne.read_epochs(os.path.join(TF, "{}_stimulus_tf_epo.fif".format(SUB_ID)))
    else:
        epoch = mne.read_epochs(os.path.join(TF, "{}_response_tf_epo.fif".format(SUB_ID)))
    epochs_list.append(epoch)

# glue all epochs together into one big Epochs object
print("\n*****\nConcatenate all Epoch objects\n*****\n")
epoch_all = mne.concatenate_epochs(epochs_list)

del epoch, epochs_list, full_epochs

#%%

# replace the spaces in the metadata titles with _ for index purposes
metadata = epoch_all.metadata
for name in metadata.columns:
    metadata = metadata.rename(columns = {name: name.replace(" ","_")})
epoch_all.metadata = metadata

del metadata, name

#%%

"""
Fast timescale: alpha and beta band activation relevant
Topoplot for repetition = 1 vs. repetition 8
Note that we work with the evoked power (first take the average of the data)
"""

print("\n******")
print("{} timescale: topoplot of Evoked data".format(TIMESCALE))
print("******\n")

# create a placeholder for the figure
fig, ax = plt.subplots(ncols   = 3, 
                       figsize = (8, 4),
                       sharex  = True, 
                       sharey  = True)

# get the data for a specific condition
f = epoch_all[COND + " == 1"].average()
l = epoch_all[COND + " == 8"].average()

# topoplots based on time
diff = mne.combine_evoked([f,l], [-1, -1])

# create three topoplots for the Evoked data
for i, pl in enumerate([f,l, diff]):
    pl.plot_topomap(TIME, 
                    ch_type      = 'eeg',
                    cmap         = "RdBu_r",
                    show_names   = True, 
                    colorbar     = False,
                    res          = 128, 
                    size         = 6,
                    average      = WINDOW,
                    time_unit    = 's',
                    contours     = 4, 
                    image_interp = "hanning",
                    axes         = ax[i])
    ax[i].set_title(TITLE[i], fontweight = "bold")
fig.suptitle("Topoplots for the Evoked data", fontweight = "bold", fontsize = 24)

del i, diff, fig, ax, pl

#%%

"""
Compute the PSD plot and get an idea of the topography
"""

print("\n******")
print("{} timescale: PSD topoplot".format(TIMESCALE))
print("******\n")

# take the first value on this timescale (rep. 1 or block 1)
e = epoch_all[COND + " == 1"]

# compute the PSD topography for this point on time
f = e.plot_psd_topomap(bands     = BAND,
                       ch_type   = "eeg",
                       tmin      = TMIN,
                       tmax      = TMAX,
                       dB        = False,
                       adaptive  = True,
                       low_bias  = True,
                       cmap      = "RdBu_r",
                       n_jobs    = 4,
                       outlines  = "head",
                       cbar_fmt  = fmt,
                       vlim      = VLIM,
                       verbose   = True)

f.tight_layout()

#del e

#%%

"""
Match the data to allow computing the difference wave by hand
"""

# load the data
f, l = epoch_all[COND + " == 1"], \
       epoch_all[COND + " == 8"]
       
# drop the external channels and the stimulus channel
f.drop_channels(['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
                 'STI 014'])
l.drop_channels(['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7','EXG8',
                 'STI 014'])

# these dimensions are used to match the Epochs when comparing 1 with 8
s1,s2 ,s3,s4,s5 = DIMS_TO_MATCH

# loop over the shortest metadata, considering that len of metadata not equal
assert len(f) !=  len(l)
m1, m2 = f.metadata, l.metadata   

# if l < f, switch what m1, m2, l and f mean to get the code below right
if len(l) < len(f):
    m1, m2 = l.metadata, f.metadata
    l, f   = f, l

# drop the indices: reset the count to 0
m1.reset_index()
m2.reset_index()

# keep track of which epochs are dropped
m1_dropped = np.repeat(True, len(m1))
m2_dropped = np.repeat(True, len(m2))

# store the indices of the matches in a separate list
paired_eps = []

# loop over the rows of the metadata, and search in each condition for data
for i in range(len(m1)):
    # save the values for each relevant variable
    vals    = []
    for j, d in enumerate(DIMS_TO_MATCH):
        vals.append(m1.iloc[i][d])
    v1,v2,v3,v4,v5 = vals
       
    # check where in the longer epoch set a list can be found
    mask = (m2[s1]==v1) & (m2[s2]==v2) & (m2[s3]==v3) & (m2[s4]==v4) & (m2[s5]==v5)
    indx = np.where(mask)
    
    # mark the indices in both lists that can be matches
    # store their indices for later array manipulation
    if len(indx[0]) == 1:
        m1_dropped[i]    = False
        m2_dropped[indx] = False
        paired_eps.append((i, indx[0][0]))

# make sure that epoch lists have same length after matching + printing
assert (len(m1) - np.sum(m1_dropped)) == (len(m2) - np.sum(m2_dropped))
print("Dropped in M1: {}\nDropped in M2: {}".format(np.sum(m1_dropped), 
                                                    np.sum(m2_dropped)))
print("Amount of matched epochs found: {}".format(len(paired_eps)))

# OPTIONAL: drop the Epochs that could not be matched in both conditions
# =============================================================================
# f.drop(indices = m1_dropped)
# l.drop(indices = m2_dropped)
# =============================================================================

del m1, m2, m2_dropped, i, j, d, mask, indx, vals, epoch_all
del s1, s2, s3, s4, s5, v1, v2, v3, v4, v5

#%%

"""
Compute the difference wave by hand
"""

# create a placeholder to store the data
diff_arr = np.zeros((len(paired_eps), 
                     len(f.info["ch_names"]),
                     len(f.times)))

# get the arrays to compute the difference
f_dat, l_dat = f.get_data(), l.get_data()

# create the events array, and the data array
for i in range(len(paired_eps)):
    i1, i2 = paired_eps[i]
    diff_arr[i] = f_dat[i1,:,:] - l_dat[i2,:,:]
  
# make a copy of f to store the data in, and drop the Epochs
# make sure the dimensions check out
diff_ep = f.copy()
diff_ep.drop(m1_dropped)
assert len(diff_ep) == len(diff_arr)

# make a copy of the original data in f (where some Epochs were dropped)
diff_ep_orig = diff_ep.get_data()
assert np.all(diff_ep_orig != diff_arr)

print("\n***\nData successfully modified")

# assign diff_arr to the diff_ep
diff_ep._data = diff_arr

# make sure that the data was modified, compare with previous, check type
assert np.all(diff_ep.get_data() == diff_arr)
assert np.all(diff_ep.get_data() != diff_ep_orig)
assert isinstance(diff_ep, mne.epochs.BaseEpochs)

del f, l, m1_dropped, diff_arr, diff_ep_orig, f_dat, l_dat, i1, i2, paired_eps

print("Modification check completed\nType check completed\n***\n")

print("******")
print("{} timescale: PSD topoplot".format(TIMESCALE))
print("******\n")

# create the power spectral density topomap for the repetition 1 - repetition 8
diff_ep.plot_psd_topomap(bands     = BAND,
                         ch_type   = "eeg",
                         tmin      = TMIN,
                         tmax      = TMAX,
                         dB        = False,
                         adaptive  = True,
                         low_bias  = True,
                         cmap      = "RdBu_r",
                         n_jobs    = 4,
                         outlines  = "head",
                         cbar_fmt  = fmt,
                         verbose   = True)

#%%

# load average TFR data
DATA    = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Plots\Permutation results\Contrasts\Data"
avg_tfr = mne.time_frequency.read_tfrs(os.path.join(DATA, TFR))[0]

# get the frequency bands
FMIN, FMAX, FNAME = BAND[0]

# plot the data
if not FAST:
    TMIN = 0
avg_tfr.plot_topomap(tmin     = TMIN,
                     tmax     = TMAX,
                     fmin     = FMIN,
                     fmax     = FMAX,
                     ch_type  = "eeg", 
                     cmap     = "RdBu_r", 
                     res      = 250,
                     outlines = "skirt", 
                     contours = 10,
                     colorbar = True,
                     cbar_fmt = fmt,
                     sensors  = "ko",
                     title    = None)
