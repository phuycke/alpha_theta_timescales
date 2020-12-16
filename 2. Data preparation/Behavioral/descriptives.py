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

import os
import pandas            as pd

#%%

# ----------- #
# DEFINE DATA #
# ----------- #

DATA = r'C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\code\Analysis\Behavioral\General\Data'
FIG  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\code\Analysis\Behavioral\Descriptive\Plots'

# keep only the Epochs for which we have the full data
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

kept = [(int(i[-2:]) - 1) for i in full_epochs]

# sex of the subjects, grouped per 5
sex = ['F', 'F', 'F', 'F', 'M',
       'M', 'F', 'F', 'F', 'F',
       'F', 'M', 'F', 'F', 'M',
       'F', 'F', 'M', 'F', 'F',
       'F', 'F', 'M', 'F', 'F', 
       'F', 'F', 'F', 'M', 'F']
sex = [sex[i] for i in kept]

# age of the subjects, grouped per 5
age = [24, 22, 22, 26, 47, 
       27, 35, 18, 19, 27,
       21, 19, 22, 19, 22,
       18, 22, 22, 21, 22,
       21, 23, 18, 26, 20,
       19, 23, 26, 24, 24]
age = [age[i] for i in kept]

# handedness of the subjects, grouped per 5
dexterity = ['R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'R',
             'L', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'R', 'L',
             'R', 'R', 'R', 'R', 'R',
             'R', 'R', 'R', 'L', 'R']
dexterity = [dexterity[i] for i in kept]

# Check for inconcistencies
if len(sex) != len(age) != len(dexterity):
    raise ValueError('The data arrays do not have the length')

# convert to Pandas dataframe
df = pd.DataFrame(list(zip(full_epochs, sex, age, dexterity)), 
                  columns = ['Subject',
                             'Sex', 
                             'Age', 
                             'Handedness']) 
    
del age, dexterity, sex

#%%

# ------------- #
# SAVE THE DATA #
# ------------- #

df.to_pickle(os.path.join(DATA, "descriptives.pkl"))
