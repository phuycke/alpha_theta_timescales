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

import numpy  as np
import os
import pandas as pd

#%%

# -------- #
# PATH DEF #
# -------- #

# the home directory where two subfolders are located: code
general  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019'
specific = r'\overtraining - PILOT 3'
path     = general + specific

# GENERAL: these directories should already exist
DATA     = path + r'\code\Analysis\Behavioral\General\Data'

del general, specific, path

#%%

# DatFrame parameters
n_sub           = 30
cleaning_params = 3

# create the dataframe: form + column names
dataframe    = np.empty((n_sub, cleaning_params)) * np.NaN
df           = pd.DataFrame(dataframe)
df.columns   = ['Bad channels', 'ICA seed used', 'Removed ICA components']

#%%

# define the bad data
bad_elecs  = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,       
              ['P9'],                                       # sub 06
              np.NaN,                                       # sub 07 (FT7 for response-locked)
              ['P7'],                                       # sub 08 (FT8 for response-locked)
              ['P2', 'PO3'],                                # sub 09
              ['F6', 'AF8'],                                # sub 10
              ['P2','PO8','Oz', 'P6'],                      # sub 11
              ['Oz', 'PO4', 'PO3'],                         # sub 12
              ['P6', 'TP8', 'P10'],                         # sub 13
              ['P7', 'PO3', 'P2', 'O2', 'P10'],             # sub 14 (bad sub)
              ['Oz','POz', 'PO4', 'PO3', 'P2', 'P7'],       # sub 15
              ['P9'],                                       # sub 16 (nice)
              np.NaN,                                       # sub 17 (very nice)
              ['TP7', 'TP8', 'Oz', 'P7', 'POz'],            # sub 18
              ['PO7', 'O1', 'P9', 'P10'],                   # sub 19
              ['P8', 'P9', 'FC6'],                          # sub 20 (nice)
              ['T8', 'TP8', 'P9'],                          # sub 21
              ['P7', 'Pz'],                                 # sub 22 
              ['P2', 'P4'],                                 # sub 23
              ['P2', 'P3', 'Iz'],                           # sub 24
              ['P8', 'O2', 'P1'],                           # sub 25
              np.NaN,                                       # sub 26
              ['P10', 'T8', 'Fp2'],                         # sub 27
              ['P7', 'P10'],                                # sub 28
              np.NaN,                                       # sub 29
              ['T7', 'T8']]                                 # sub 30

# count which chanels are interpolated 
d = dict()
for l in bad_elecs:
    uniq = np.unique(l)
    for k in uniq:
        if isinstance(k, np.str_):
            if k not in d:
                d[k] = 1
            else:
                d[k] += 1

# sort the dict by key values
for w in sorted(d, key = d.get, reverse = True):
    print(w, d[w])
    
#%%

# define the excluded ICA components (stimulus locked)
ICA_comps  = [np.NaN,                                       # sub 01
              [1],                                          # sub 02 
              [4],                                          # sub 03 
              [0],                                          # sub 04    
              [6],                                          # sub 05
              [0],                                          # sub 06
              [0],                                          # sub 07
              [0],                                          # sub 08
              [3],                                          # sub 09
              [0],                                          # sub 10
              [0],                                          # sub 11
              [2],                                          # sub 12
              [0],                                          # sub 13
              np.NaN,                                       # sub 14
              [0],                                          # sub 15
              [1],                                          # sub 16
              [0],                                          # sub 17
              [0],                                          # sub 18
              [0],                                          # sub 19
              [0],                                          # sub 20
              [0],                                          # sub 21
              [0],                                          # sub 22
              [1],                                          # sub 23
              [1],                                          # sub 24
              [5],                                          # sub 25
              [0],                                          # sub 26
              [2],                                          # sub 27
              [0],                                          # sub 28
              [4],                                          # sub 29
              [0]]                                          # sub 30


# define the excluded ICA components (response-locked)
ICA_comps  = [np.NaN,                                       # sub 01
              [1],                                          # sub 02 
              [4],                                          # sub 03 
              [0],                                          # sub 04    
              [6],                                          # sub 05
              [0],                                          # sub 06
              [0],                                          # sub 07
              [0],                                          # sub 08
              [3],                                          # sub 09
              [0],                                          # sub 10
              [0],                                          # sub 11
              [2],                                          # sub 12
              [0],                                          # sub 13
              np.NaN,                                       # sub 14
              [0],                                          # sub 15
              [1],                                          # sub 16
              [0],                                          # sub 17
              [0],                                          # sub 18
              [0],                                          # sub 19
              [0],                                          # sub 20
              [0],                                          # sub 21
              [0],                                          # sub 22
              [1],                                          # sub 23
              [1],                                          # sub 24
              [5],                                          # sub 25
              [0],                                          # sub 26
              [2],                                          # sub 27
              [0],                                          # sub 28
              [4],                                          # sub 29
              [0]]                                          # sub 30

#%%

# define the random seed used for the ICA
seed_used  = [np.NaN] + ([2019] * (n_sub - 1))

#%%

del n_sub, cleaning_params, dataframe

# write the data to a pandas DataFame
df['Bad channels'].iloc[0:len(bad_elecs)]           = np.array(bad_elecs, dtype=object)
df['ICA seed used'].iloc[0:len(seed_used)]          = np.array(seed_used, dtype=object)
df['Removed ICA components'].iloc[0:len(ICA_comps)] = np.array(ICA_comps, dtype=object)

del bad_elecs, seed_used, ICA_comps

#%%

# ------------------------------- #
# SAVE ALTERED DATA FOR LATER USE #
# ------------------------------- #

df.to_pickle(os.path.join(DATA, "eeg_cleaning_params.pkl"))
