#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

"""
Globals
Variables that will be used throughout the script. Marked by all capitals in 
line with convention in the C language.
"""

# Determines which part of the analysis to run
STIM_LOCKED = True
CHOICE = "untrans"

# these are the subjects that had all 512 epochs recorded and stored safely
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

#%%

import mne
import numpy    as np
import os
from matplotlib import rcParams

#%%

if False:
    # the root folder, but split up for readability
    general  = r"C:\Users\pieter\OneDrive - UGent\Projects\2019"
    specific = r"\overtraining - PILOT 3"
    path     = general + specific
    
    DATA     = os.path.join(path, "figures", "TF", "Group level", "data")
    FIG      = os.path.join(path, "figures", "TF", "Group level", "plots")
    
    del general, specific
    
    
    """
    Subject specific data
    Here we store all the epoch data for each subject in a list. This was done so
    that the data for each subject can be accessed to do e.g. TF on this data. The
    data is also merged into one single Epoch file, which can be used to do an 
    overall analysis (e.g. the TF representation of block 1 over subjects).
    """
    
    epochs_list = []
    
    # get a list of all possible subject numbers, and loop over them
    for SUB_ID in full_epochs:     
    
        # path depends on the subject ID, hence no global def
        ROOT  = path + r"\{}\ses-01\eeg\Python".format(SUB_ID)
        TF    = os.path.join(ROOT, "time-frequency")
            
        # skip sub-14 as this is a bad subject
        if SUB_ID == "sub-14":
            continue
        
        # read the relevant epoch data and store in epochs_list
        if STIM_LOCKED:
            epoch = mne.read_epochs(os.path.join(TF, "{}_stimulus_tf_epo.fif".format(SUB_ID)))
        else:
            epoch = mne.read_epochs(os.path.join(TF, "{}_response_tf_epo.fif".format(SUB_ID)))
        epochs_list.append(epoch)
    
    # glue all epochs together into one big Epochs object
    print("Concatenate all Epoch objects")
    epoch_all = mne.concatenate_epochs(epochs_list)
    
    # wavelet parameters
    min_freq = 4
    max_freq = 30
    num_frex = 15
    
    # define frequencies of interest
    frequencies = np.logspace(np.log10(min_freq), 
                              np.log10(max_freq), 
                              num_frex)
    
    del min_freq, max_freq, num_frex
        
    """
    Comparing condition x with condition y
    In this code block, we compare several TF representations. Specifically, we 
    compare the TF of x with a, b, c, ... d. Here, this means that we compare
    block 8 with block 1, block 2, ... block 7. The TF of x is plotted on the 
    lower part of the plot, the others go right above.
    """
    
    # replace the spaces in the metadata titles with _
    metadata = epoch_all.metadata
    for name in metadata.columns:
        metadata = metadata.rename(columns = {name: name.replace(" ","_")})
    epoch_all.metadata = metadata
        
    # stim/resp locked epochs: fast timescale first timepoint
    block_1 = mne.time_frequency.tfr_morlet(epoch_all["Block_number_specific == 1"], 
                                            picks      = "eeg",
                                            freqs      = frequencies,
                                            n_cycles   = frequencies / 2,   # 6
                                            return_itc = False, 
                                            verbose    = True,
                                            n_jobs     = 4)
    
    # stim/resp locked epochs: fast timescale last timepoint
    block_8 = mne.time_frequency.tfr_morlet(epoch_all["Block_number_specific == 8"], 
                                            picks      = "eeg",
                                            freqs      = frequencies,
                                            n_cycles   = frequencies / 2,   # 6
                                            return_itc = False, 
                                            verbose    = True,
                                            n_jobs     = 4)
    
    # save the data
    block_1.save(fname = r"C:\Users\pieter\Downloads\block 1 (24 subs)-tfr.h5")
    block_8.save(fname = r"C:\Users\pieter\Downloads\block 8 (24 subs)-tfr.h5")

#%%

# load the TFR data
block1 = mne.time_frequency.read_tfrs(r"C:\Users\pieter\Downloads\block 1 (24 subs)-tfr.h5")[0]
block8 = mne.time_frequency.read_tfrs(r"C:\Users\pieter\Downloads\block 8 (24 subs)-tfr.h5")[0]
    
# transform the data
options = ["untrans", "log_sub", "log_div"]
assert CHOICE in options

# transforms the data in different ways
if CHOICE == "untrans":
    # save block8 in temp, en dan subtract block1 from the data in temp
    temp = block8
    temp._data -= block1._data
    
    # check whether the difference does not equal rep_1 or rep_8
    assert np.all(temp._data != block1._data)
    assert not np.sum(temp._data != block8._data)
    
    # colorscale args
    VMIN, VMAX = None, None
    
elif CHOICE == "log_sub":
    # save block8 in temp, dB transform
    temp = block8
    temp._data = 10 * np.log10(block8._data)
    
    # save block1 in temp2, dB transform
    temp2 = block1
    temp2._data = 10 * np.log10(block1._data)
    
    temp._data -= temp2._data
        
    # check whether the difference does not equal rep_1 or rep_8
    assert np.all(temp._data != block1._data)
    assert not np.sum(temp._data != block8._data)
    
    VMIN, VMAX = 0, 5
    
elif CHOICE == "log_div":
    # save block8 in temp, dB transform
    temp  = block8
    temp2 = block1
    
    temp._data -= 10 * np.log10(temp._data / temp2._data)
        
    # check whether the difference does not equal rep_1 or rep_8
    assert np.all(temp._data != block1._data)
    assert not np.sum(temp._data != block8._data)
    
    VMIN, VMAX = None, None
    
#%%

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size']   = 30

#%%

"""
Globals
Variables that will be used throughout the script. Marked by all capitals in 
line with convention in the C language.
"""

BAND          = [(4, 8, "Theta")]
TMIN, TMAX    = 0, .4

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

# define the data
avg_tfr = temp

# get the frequency bands
FMIN, FMAX, FNAME = BAND[0]

# make topoplot
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
