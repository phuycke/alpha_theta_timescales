#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

# import modules
import numpy as np
import os

root = r'C:\Users\pieter\Downloads\GitHub\Pieter_H\2018-2019\projects\ruge\extensive training - experiment\overtraining'
os.chdir(root + '\code\experiment')

from   randomizationHelper import randomHelper as rh

#%%

# ----------- #
# DEFINITIONS #
# ----------- #

mappings           = 9
number_of_subjects = 30

subject_numbers    = np.arange(1, number_of_subjects + 1)
mapping_numbers    = np.arange(1, mappings + 1)

#%%

# make the arrays for all 30 participants
conditions = np.split(np.zeros(number_of_subjects * 2), 
                      number_of_subjects)

# three mappings keep returning
subsample  = np.random.choice(mapping_numbers, 
                              3, 
                              replace=False)

print("\nLOG: the following mappings are returning:")
for i in range(len(subsample)):
    print("\tMapping: {}".format(subsample[i]))

# determine which mappings keep returning
returning  = np.repeat(subsample, 
                       number_of_subjects / 3)

for i in range(number_of_subjects):
    conditions[i][0] = returning[i]
    
# determine the fixation cross that is used
    # 0 equals blue  for focus, black for relax
    # 1 equals black for focus, blue  for relax
fixation   = np.split(np.tile([0, 1], number_of_subjects // 2), 
                      3)

for j in range(len(fixation)):
    np.random.shuffle(fixation[j])

fixation = np.ravel(fixation)
for k in range(len(fixation)):
    conditions[k][1] = fixation[k]
    
#%%
    
"""
shuffle trials
the number that keeps returning represents the mapping that the participants 
keep seeing
"""
row_length = (mappings - 1) * 2
trials     = np.zeros((number_of_subjects,
                       row_length))

for i in range(trials.shape[0]):
    searching = True
    while searching:
        remaining    = np.delete(mapping_numbers, np.where(mapping_numbers == conditions[i][0]))
        np.random.shuffle(remaining)
        possible_row = np.zeros(trials[i,:].shape)
        for j in range(0, trials.shape[1], 2):
            flip = np.random.random()
            if (flip > 0.5):
                possible_row[j]   = conditions[i][0]
                possible_row[j+1] = remaining[j // 2]
            else:
                possible_row[j+1] = conditions[i][0]
                possible_row[j]   = remaining[j // 2]
        if i == 0:
            trials[i,:] = possible_row
            searching = False
        else:
            nonzero = trials[np.all(trials, axis=1)]
            if not (nonzero == possible_row).all(1).any():
                trials[i,:] = possible_row
                searching = False

if len(set(map(tuple,trials)))==len(trials):
    print('\nLOG: The resulting trial matrix consists of unique rows\n')
else:
    raise Exception('\nLOG: Randomisation failed, please rerun script')

#%%
    
for k in range(1, number_of_subjects):
    selected  = trials[k,:]
    recurrent = int(conditions[k][0])
    
    all_trials   = np.zeros((row_length * 32))
    all_trial_id = np.zeros((row_length * 32, 3))
    all_resp     = np.zeros((row_length * 32))
    
    for l in range(len(selected)):
        lb      = (int(selected[l]) - 1) * 4
        up      = int(selected[l]) * 4
        stimuli = np.arange(lb, up)
        
        recur = True
        if int(selected[l]) != recurrent:
            recur = False
            
        trial, trial_id, resp = rh.probabilistic_shuffle(stimulus_array = stimuli, 
                                                         trials         = 32, 
                                                         returning      = recur)
        
        all_trials[l*32:(l+1)*32]      = trial
        all_trial_id[l*32:(l+1)*32, :] = trial_id
        all_resp[l*32:(l+1)*32]        = resp
    
    # update stimulus repetition count
    stimuli  = np.unique(all_trial_id[:,0])                 # make a list of all stimuli
    counted  = np.zeros(len(stimuli))                       # make a list with occurences, one for each stimulus
    
    for m in range(len(all_trial_id[:,1])):                 # loop over all trials
        stim              = all_trial_id[m,0]               # the stimulus
        indx              = np.where(stimuli == stim)[0]    # find place in occurences that needs updating
        counted[indx]    += 1                               # update occurences
        all_trial_id[m,1] = counted[indx]                   # place updated value in data array
    print('LOG: Subject {0:02d}: stimulus count updated'.format(k + 1))
        
    os.chdir(root + r'\sub-{0:02d}'.format(k + 1))
    
    # save data
    np.save('sub-{0:02d}-rand param.npy'.format(k + 1), conditions)
    np.save('sub-{0:02d}-trial.npy'.format(k + 1), all_trials)
    np.save('sub-{0:02d}-trial id.npy'.format(k + 1), all_trial_id)
    np.save('sub-{0:02d}-response.npy'.format(k + 1), all_resp)

#%%
print('\nLOG: Randomisation complete\nLOG: All data written away safely')
