#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

import mne
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
BAND          = [(8, 12, "Alpha")]
TMIN, TMAX    = .65, .9  
TFR           = "stim-locked repetition 8 - repetition 1 (non phase)-tfr.h5"

# the root folder, but split up for readability
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3"
DATA = os.path.join(ROOT, "figures", "TF", "Group level", "data")

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

# load average TFR data
DATA    = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Plots\Permutation results\Contrasts\Data"
avg_tfr = mne.time_frequency.read_tfrs(os.path.join(DATA, TFR))[0]

# get the frequency bands
FMIN, FMAX, FNAME = BAND[0]

# plot the data
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
