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
import numpy    as np
import os

from matplotlib import rcParams

#%%

"""
Globals
Variables that will be used throughout the script. Marked by all capitals in 
line with convention in the C language.
"""

# Determines which part of the analysis to run + some plotting parameters
STIM_LOCKED = True
COMPUTE_TFR = False
BAND        = [(4, 8, "Theta")]
TMIN, TMAX  = 0, .4
VMIN, VMAX  = -.85, .85

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size']   = 30

# these are the subjects that had all 512 epochs recorded and stored safely
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

#%%

# bool is set to True, you can calculate the difference yourself using this code
if COMPUTE_TFR:
    # the root folder, but split up for readability
    path  = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3"   
    
    # holder for all epochs
    epochs_list = []
    
    # get a list of all possible subject numbers, and loop over them
    for SUB_ID in full_epochs:     
    
        # path depends on the subject ID, hence no global def
        ROOT  = path + r"\{}\ses-01\eeg\Python".format(SUB_ID)
        TF    = os.path.join(ROOT, "time-frequency")
            
        # read the relevant epoch data and store in epochs_list
        if STIM_LOCKED:
            epoch = mne.read_epochs(os.path.join(TF, "{}_stimulus_tf_epo.fif".format(SUB_ID)))
        else:
            epoch = mne.read_epochs(os.path.join(TF, "{}_response_tf_epo.fif".format(SUB_ID)))
        epochs_list.append(epoch)
    
    # glue all epochs together into one big Epochs object
    print("Concatenate all Epoch objects")
    epoch_all = mne.concatenate_epochs(epochs_list)
    
    # define frequencies of interest
    frequencies = np.logspace(np.log10(4), 
                              np.log10(30), 
                              15)      
   
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

# save block8 in temp, dB transform
temp = block8
temp._data = 10 * np.log10(block8._data)

# save block1 in temp2, dB transform
temp2 = block1
temp2._data = 10 * np.log10(block1._data)

temp._data -= temp2._data
    
# check whether the difference does not equal block1 or block8
assert np.all(temp._data != block1._data)
assert not np.sum(temp._data != block8._data)

#%%

# colorbar with log scaled labels
def fmt_float(x, pos):
    return r'${0:.2f}$'.format(x)

#%%

# define the data
avg_tfr = temp

# get the frequency bands
FMIN, FMAX, FNAME = BAND[0]

# create an axis to plot the figure on
fig, ax = plt.subplots()

# make topoplot
avg_tfr.plot_topomap(tmin     = TMIN,
                     tmax     = TMAX,
                     fmin     = FMIN,
                     fmax     = FMAX,
                     vmin     = VMIN, 
                     vmax     = VMAX,
                     ch_type  = "eeg", 
                     cmap     = "RdBu_r",
                     outlines = "head", 
                     contours = 10,
                     colorbar = True,
                     cbar_fmt = fmt_float,
                     sensors  = "ko",
                     axes     = ax,
                     title    = None)

# set the Figure dir + set the size of the image
FIG = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Correct DPI plots"
fig.set_size_inches(6, 4)

# save as tiff and pdf
plt.savefig(fname = os.path.join(FIG, "Figure 4B.tiff"), dpi = 300)
plt.savefig(fname = os.path.join(FIG, "Figure 4B.tiff"), dpi = 300)

plt.close("all")