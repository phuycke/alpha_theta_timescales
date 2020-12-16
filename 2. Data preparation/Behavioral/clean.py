#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

# -------------- #
# IMPORT MODULES #
# -------------- #

import mne
import numpy             as np
import os
import warnings

#%%

# hide the warnings for the Matplotlib module (thrown for colorbar def)
warnings.filterwarnings("ignore", 
                        category = UserWarning, 
                        module   = 'matplotlib')

#%%

# ---------------- #
# SOME DEFINITIONS #
# ---------------- #

# the root folder, but split up for readability
general  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019'
specific = r'\overtraining - PILOT 3'
path     = general + specific

del general, specific

#%%

min_del = max_del = 0

# these are the subjects that had all 512 epochs recorded and stored safely
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

# store how much was deleted
deleted = np.zeros(len(full_epochs))
i = 0

# get a list of all possible subject numbers, and loop over them
for SUB_ID in full_epochs:     
       
    # the path where the data is stored
    ROOT  = path + r'\{}\ses-01\eeg\Python'.format(SUB_ID)
    EPOCH = os.path.join(ROOT, 'epoch', '{}_stimulus_epo.fif'.format(SUB_ID))
    TF    = os.path.join(ROOT, 'time-frequency')
             
    print('\n- - -\nProcessing {}\n- - '.format(SUB_ID))
    
    # load original data and the metadata
    epoch    = mne.read_epochs(EPOCH)
    metadata = epoch.metadata
    
    # replace the spaces in the metadata titles with _
    for name in metadata.columns:
        metadata = metadata.rename(columns = {name: name.replace(" ","_")})

    # reassign the metadata
    epoch.metadata = metadata
    
    # ---------------- #
    # CLEAN DATA (1)   # 
    # REMOVE WEIRD RTs #
    # ---------------- #
    
    # delete trials where the RT equals 0
    epoch = epoch['Reaction_times != 0']
    
    print('\n{0}:\nRemove trials where RT = 0.'.format(SUB_ID))
    
    # filter data based on the reaction time (remove RT outliers)
    rt_median = np.median(epoch.metadata['Reaction_times'])
    rt_sd     = np.std(epoch.metadata['Reaction_times'])
    
    dim_orig = len(epoch)
    epoch    = epoch['Reaction_times >= {0} and Reaction_times <= {1}'.format(rt_median - 3 * rt_sd, 
                                                                              rt_median + 3 * rt_sd)]
    print('\n{0}:\n{1:.2f}% of the data deleted (deviation from mean).'.format(SUB_ID,
                                                                              (dim_orig - len(epoch))/(dim_orig/100)))
    del rt_median, rt_sd
    
    # ------------------------ #
    # CLEAN DATA (2)           # 
    # REMOVE ATTENTIONAL SLIPS #
    # ------------------------ #
    
    # reassign the metadata var
    metadata = epoch.metadata
    
    # remove trials where subjects are wrong for stimuli seen at least 3 times
    dim_orig = len(epoch)
    slips    = metadata[(metadata['In_block_repetitions'] >= 3) & (metadata['Correct'] == 0)]
    epoch    = epoch[~metadata.index.isin(slips.index)]
    
    print('\n{0}:\n{1:.2f}% of the data deleted (attentional slips).'.format(SUB_ID,
                                                                            (dim_orig - len(epoch))/(dim_orig/100)))
    del slips, dim_orig
    
    # ------------------ #
    # CLEAN DATA (3)     # 
    # ADD SUBJECT NUMBER #
    # ------------------ #
       
    # reassign the metadata var
    metadata = epoch.metadata
    
    # add the subject number to the metadata
    epoch.metadata['Subject number'] = [int(SUB_ID[-2:])] * len(epoch.metadata)
    
    # reorder the columns of the df
    df_cols   = epoch.metadata.columns.tolist()
    new_order = [df_cols[-1]]
    [new_order.append(col) for col in df_cols[:-1]]
    epoch.metadata = epoch.metadata[new_order]
    
    # remove the trailing _ in the col names
    for name in metadata.columns:
        metadata = metadata.rename(columns = {name: name.replace("_"," ")})
    epoch.metadata = metadata
    
    # ----------------- #
    # SAVE CLEANED DATA #
    # ----------------- #
    
    # save the reprocessed data in the specific directory
    epoch.save(os.path.join(TF, '{}_stimulus_epo.fif'.format(SUB_ID)), 
               overwrite = True, 
               verbose   = True)
    
    # reassign the metadata var
    deleted[i] = len(metadata) / 5.12
    i += 1