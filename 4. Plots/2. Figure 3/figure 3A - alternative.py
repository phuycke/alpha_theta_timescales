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

#%%

# the root folder, but split up for readability
general  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019'
specific = r'\overtraining - PILOT 3'
path     = general + specific

DATA     = os.path.join(path, 'figures', 'TF', 'Group level', 'data')
FIG      = os.path.join(path, 'figures', 'TF', 'Group level', 'plots')

del general, specific

#%%

"""
Subject specific data
Here we store all the epoch data for each subject in a list. This was done so
that the data for each subject can be accessed to do e.g. TF on this data. The
data is also merged into one single Epoch file, which can be used to do an 
overall analysis (e.g. the TF representation of block 1 over subjects).
"""

epochs_list = []

# get a list of all possible subject numbers, and loop over them
for SUB_ID in ['sub-{0:02d}'.format(i) for i in range(2, 31)]:     

    # path depends on the subject ID, hence no global def
    ROOT  = path + r'\{}\ses-01\eeg\Python'.format(SUB_ID)
    TF    = os.path.join(ROOT, 'time-frequency')
        
    # skip sub-14 as this is a bad subject
    if SUB_ID == 'sub-14':
        continue
    
    # read the relevant epoch data and store in epochs_list
    epoch = mne.read_epochs(os.path.join(TF, '{}_stimulus_tf_epo.fif'.format(SUB_ID)))
    epochs_list.append(epoch)

#%%

# wavelet parameters
min_freq = 4
max_freq = 30
num_frex = 15

# define frequencies of interest
frequencies = np.logspace(np.log10(min_freq), 
                          np.log10(max_freq), 
                          num_frex)

del min_freq, max_freq, num_frex


#%%

'''
Placeholder
'''

# compute AverageTFR to store the F values in later on
placeholder = mne.time_frequency.tfr_morlet(epochs_list[0], 
                                            picks      = "eeg",
                                            freqs      = frequencies,
                                            n_cycles   = frequencies / 2,
                                            return_itc = False, 
                                            verbose    = True,
                                            n_jobs     = 4)

#%%

# load the data, and store the F values in placeholder
PERM_DATA = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Repetition 1 vs. repetition 8"
placeholder._data = np.load(os.path.join(PERM_DATA, "f_obs.npy"))

# adjust time to make sure that you only plot between 0 and 1 seconds
placeholder.times = placeholder.times[np.where((placeholder.times > 0) & (placeholder.times <= 1))[0]]

# plot
placeholder.plot(baseline = (None, None),
                 yscale   = 'log',
                 combine  = 'mean')
