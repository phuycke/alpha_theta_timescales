#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

import numpy as np
import os

#%%
 
   
class randomHelper:
    

    def __init__(self, number_of_conditions):
        self.num_cond = number_of_conditions


    def dataHandler(path = None, 
                    subs_id = None,
                    subs_total = None):
        
        """
        :param path:
        The amount of unique stimuli that are shown within each experiment block
    
        :param subs_id:
        The specific subject ID, used to cd to the relevant subject-specific
        directory
             
        :param subs_total:
        The total amount of subjects, used to create a correct amount of 
        subject-specific subfolders in your 'subjects' root folder
        
        :return:
        What is returned depends on the usage:
            - a folder structure for data storage
            - a change of directory to a subject-specific folder
        
        If path is not defined, an error will be raised
        If you want to create a folder structure to store your data:
            - assign a value to 'subs_total', but not to 'subs_id'
        If you want to cd to a relevant directory, e.g. to access stored data:
            - assign a value to 'subs_id', but not to 'subs_total'
        
        :example use:
        dataHandler(path       = directory/where/data/is/stored,
                    subs_total = 20)
        
        This will create a root folder named 'subjects', and 20 subfolders:
            - subject  1
            - subject  2
            - ...
            - subject 20
        """
        
        if (subs_id is not None and subs_total is not None):
            raise Exception('\n!! Conflict in passed parameters !!\n'
                            'If "subs_id" is non-empty, "subs_total" cannot take a value, and vice versa.')
        
        if subs_id is None:
            if path is None:
                raise ValueError('You must define the path where your data will be stored.')
            elif not isinstance(path, str):
                raise ValueError('A string must be assigned to "path"')
            else:
                try:
                    os.chdir(path)
                except FileNotFoundError:
                    print('-- FileNotFoundError --\nThis path does not exist\nPlease provide an existing path.')
            
            # make a 'Simulations' directory within each folder
            root = 'subjects'
            if not os.path.exists(root):
                os.mkdir(root)
            else:
                pass
            
            os.chdir(root)
            if subs_total is None:
                subs_total = 20
                print('Warning: no user-defined amount of subjects\n'
                      'Warning: a total of 20 subjects is assumed')
                
            total = len(list(str(subs_total)))
    
            for i in range(1, subs_total + 1):
                sub_folder = 'subject {0:{1:d}d}'.format(i, total)
                if not os.path.exists(sub_folder):
                    os.mkdir(sub_folder)
                else:
                    pass
        else:
            if path is None:
                raise ValueError('You must define the path where your data will be stored.')
            elif not isinstance(path, str):
                raise ValueError('A string must be assigned to "path"')
            else:
                try:
                    os.chdir(path)
                except FileNotFoundError:
                    print('-- FileNotFoundError --\nThis path does not exist\nPlease provide an existing path.')
             
            # cd to the correct directory based on subject number
            try:
                os.chdir('sub-{:02d}'.format(subs_id))
            except FileNotFoundError:
                raise FileNotFoundError('\nPath not found, please run the following line:\n'
                                        'randomHelper.dataHandler(path = relevant_path, subs_total = n)')


    def latinSquare(to_balance = None, 
                     subjects = None, 
                     conditions = None):
        
        """
        :param to_balance:
        Balance an inputted array of values
        Providing a value will yield a latin square with n x n dimensions, 
        where n = len(inputted array)
    
        :param subjects:
        The amount of subjects that will be tested in your experiment
        
        :param conditions:
        The number of different conditions one has in the experiment
        
        :return:
        Returns an array containing a latin square design matrix
        type(output): NumPy array
        
        :example use:
        latinSquare(subjects = 30, conditions = 5)
            --> perfectly balanced design
        latinSquare(subjects = 32, conditions = 5)
            --> non-balanced design
        latinSquare(to_balance = np.arange(1, 6))
            --> will make a latin square from this: array([1, 2, 3, 4, 5])
        latinSquare(to_balance = np.arange(1, 6), subjects = 10)
            --> will throw an Exception, as these two parameters cannot exist together   
        """
        
        if (to_balance is not None) and (subjects is not None or conditions is not None):
            raise Exception('\nConflicting input, please review your inputted parameters.\n'
                            'If to_balance is non-empty, the other arguments cannot take a value.')
        
        if to_balance is None:            
            if subjects is None:
                num_sub = 20
            else:
                num_sub = subjects
            
            if conditions is None:
                num_cond = 5
            else:
                num_cond = conditions
                 
            indx          = 1
            all_trials    = np.zeros((num_sub, num_cond))
            all_trials[0] = np.arange(1, num_cond+1)
            
            while indx != num_sub:
                
                # array of all zeros, n x n dimensions, with n the number of conditions
                start_array = all_trials[indx - 1]
                begin       = start_array[:1]
                end         = start_array[1:]
                all_trials[indx] = np.concatenate((end, begin), axis = None)
                
                indx += 1
            
            # return the output
            return all_trials.reshape((num_sub, num_cond))
        else:
            if type(to_balance) is not np.ndarray:
                raise ValueError('The provided example should be a NumPy array')
            
            indx          = 1
            all_trials    = np.zeros((len(to_balance), len(to_balance)))
            all_trials[0] = to_balance
            
            while indx != len(to_balance):
                
                # array of all zeros, n x n dimensions, with n the number of conditions
                start_array = all_trials[indx - 1]
                begin       = start_array[:1]
                end         = start_array[1:]
                all_trials[indx] = np.concatenate((end, begin), axis = None)
                
                indx += 1
            
            # return the output
            return all_trials
    
    
    def random_row_shuffle(arr,
                           start, 
                           n_rows):
        	
        """
        :return:
        Returns an array containing a latin square design matrix
        type(output): NumPy array
        
        :example use:
        random_row_shuffle(arr    = trial_array, 
                           start  = 0, 
                           n_rows = 4)
        This will return trial_array, but the first four rows will be shuffled
        """
        
        np.random.shuffle(arr[start:start + n_rows])
    
    
    def controlled_row_shuffle(to_shuffle, 
                               n_rows = 4, 
                               n_cols = 24,
                               controlled = True):
        
        """
        :param to_shuffle:
        the array that one want to shuffle row-wise
        type(to_shuffle) = NumPy array
    
        :param n_rows:
        The amount of rows you want to shuffle
        
        :param conditions:
        In the controlled version, you might want to check values up to a 
        certain column
        The parameter n_cols might help in this
        
        :param controlled:
        Used in this experiment, makes sure that participants do not see the
        same stimulus twice in a row
        
        :return:
        Returns an array
        Interpretation depends on the use of the function
        
        :example use:
        controlled_row_shuffle(array)
            --> returns an array where each 4 rows are shuffled
            --> we limit ourselves to check the first 24 columns
        """
        
        indices  = np.arange(0, len(to_shuffle), n_rows)
        
        if not controlled:
            for indx in indices:
                randomHelper.random_row_shuffle(to_shuffle, indx, n_rows)
        else:
            for i in range(len(indices)):
                if i == 0:
                    randomHelper.random_row_shuffle(to_shuffle, indices[i], n_rows)
                else:
                    same_index = True
                    while same_index:
                        prev_indx = np.argmax(to_shuffle[indices[i]-1:indices[i],:n_cols])
                        randomHelper.random_row_shuffle(to_shuffle, indices[i], n_rows)
                        curr_indx = np.argmax(to_shuffle[indices[i]:indices[i]+1,:n_cols])
                        if prev_indx != curr_indx:
                            same_index = False
        
        trials_shuff   = to_shuffle[:,:n_cols]
        trial_id_shuff = to_shuffle[:,n_cols:n_cols+3]
        expected_shuff = to_shuffle[:,n_cols+3:]
        
        return trials_shuff, trial_id_shuff, expected_shuff
    
    
    def probabilistic_shuffle(stimulus_array = None, 
                              trials         = 32,
                              returning      = False, 
                              verbose        = False):

        """
        :param stimulus_array:
        the array that one want to shuffle row-wise
        type(to_shuffle) = NumPy array
    
        :param trials:
        The amount of rows you want to shuffle
        
        :param returning:
        Indicates whether this is the returning stimulus mapping or not, is
        used to determine trial ID 
        
        :return:
        Returns an array depicting the stimuli that will be shown
        Also returns a trial ID array, and the expected response
        
        :example use:
        probabilistic_shuffle(stimulus_array=np.arange(8, 12), 
                              trials = 32, 
                              returning=True)
        --> returns an three arrays with each 32 rows
        """
    
        if type(stimulus_array) is not np.ndarray:
            raise ValueError('The provided example should be a NumPy array')
        
        # returning the shuffled stimulus presentation
        need_to_shuffle = np.copy(stimulus_array)             # make copy for manipulation
        trial_arr       = np.zeros(trials)                    # where the stimuli are stored
        probabilities   = np.full((1, len(need_to_shuffle)),  # initially set to p for all stimuli
                                   1/len(need_to_shuffle))[0]
        still_available = np.full((1, 4),                     # how many times can we still draw a stimulus
                                  trials // len(need_to_shuffle))[0]
        benchmark       = np.full((1, 4),                     # result should be this
                                  trials // len(need_to_shuffle))[0]
        counting        = 1
        
        # defining the response        
        left_resps     = need_to_shuffle[:2]
        
        response_array = np.zeros(trials)
        trial_id       = np.zeros((trials, 3))
        
        while True:
            '''
            In this loop, we draw stimulus presentations
            At the end, we check whether each stimulus was shown the same amount of times
            The process is probabilistic, but will restart if an anomaly is detected
            '''
            for i in range(trials):            
                chosen                   = np.random.choice(need_to_shuffle,
                                                            p = probabilities)
                index                    = np.where(need_to_shuffle == chosen)[0]
                
                # updating other arrays
                if chosen in left_resps:
                    response_array[i]    = 0
                else:
                    response_array[i]    = 1
                    
                trial_id[i, 0]           = chosen
                if not returning:
                    trial_id[i, 2]       = 0
                else:
                    trial_id[i, 2]       = 1
                
                still_available[index]  -= 1
                trial_arr[i]             = chosen
                trial_id[i, 1]           = 8 - still_available[index]
                if np.amin(still_available) > 0:
                    probabilities            = still_available / (trials - (i + 1))
                else:
                    non_zero_indx   = np.nonzero(still_available)[0]
                    
                    # keep non-zero elements only
                    still_available = still_available[non_zero_indx]
                    probabilities   = probabilities[non_zero_indx]
                    need_to_shuffle  = need_to_shuffle[non_zero_indx]
                    
                    # compute new probability
                    probabilities   = still_available / (trials - (i + 1))
        
            elements, uind = np.unique(trial_arr, return_inverse=True)
            occurence      = np.bincount(uind)
        
            if np.all(np.equal(occurence, benchmark)):
                if verbose:
                    print('\nLOG: combination found in cycle {0}\nLOG: process completed'.format(counting))
                break
            else:
                need_to_shuffle = stimulus_array
                still_available = np.full((1, 4), trials // len(need_to_shuffle))[0]
                probabilities   = np.full((1, len(need_to_shuffle)),
                                   1/len(need_to_shuffle))[0]
                trial_arr       = np.zeros(trials)
                counting       += 1
        
        return trial_arr, trial_id, response_array

    
    def stimulusOrder(uniq_stim   = 4,
                      no_diff_map = 6,
                      returning   = None, 
                      block_order = None):
    
        """
        :param uniq_stim:
        The amount of unique stimuli that are shown within each block of the same trialtype
    
        :param no_diff_map:
        The amount of unique S-R mappings we have in the experiment
            
        :param returning:
        If specified, indicates which mapping will be repeated
        If left undefined, the returning mapping will be drawn randomly
        Returning should be of type int or numpy.int
        
        :param block_order:
        Can be used to define in which order the mappings are presented
        If left undefined, the numerical order will be maintained
        This input should be a NumPy array
        An example:
            - stimulusOrder(returning = 2, [5. 4. 3. 1. 6.])
        Then, the order will be [SR1 - SR5] [SR1 - SR4] ... [SR1 - SR6]
        
        :return:
        Returns the order of stimulus representation, impacted by the values of 
        the function parameters.
        type(output): NumPy array
        """
        
        # definitions
        no_outputs       = 2
        trials_per_block = uniq_stim * 8 * 2
        input_units      = no_diff_map * uniq_stim   
        trial_total      = trials_per_block * (no_diff_map - 1) 
        
        # input patterns
        activation = np.zeros((input_units, input_units))
        for index in range(input_units):
            activation[index,index] = 1
        
        # for now, mapping 2 will always return
        if isinstance(returning, int) or isinstance(returning, (int, np.integer)):
            if 1 <= returning < no_diff_map + 1:
                returning -= 1
                pass
            else:
                raise ValueError('Returning should be an int within [1, {}]'.format(no_diff_map))
        elif returning is None:
            returning = np.random.randint(0, no_diff_map) 
        else:
            raise ValueError('Returning should be an int within [1, {}]'.format(no_diff_map))
        
        if block_order is None:
            # sequential block order
            block_list = np.delete(np.arange(0, no_diff_map), returning)
        else:
            if type(block_order) is not np.ndarray:
                    raise ValueError('The provided block order should be a NumPy array') 
            # prespecified block order
            block_list = np.subtract(block_order, 1).astype(int)
        
        # create a block of 32 trials with recurring S-R mapping
        index       = returning * uniq_stim
        recur_block = np.tile(activation[index:index + uniq_stim,:], (trials_per_block // 8, 1))
        answers     = np.tile(np.array([[0, 1], [1, 0]]), (recur_block.shape[0], 1))
           
        # delete recurring mapping from mapping list, and make array for inputs
        shown_patterns  = np.zeros((no_diff_map - 1, trials_per_block, input_units))
        associated_resp = np.zeros((no_diff_map - 1, trials_per_block, no_outputs))
        
        for i in range(len(block_list)):
            # specific block
            id_number      = block_list[i] * uniq_stim
            specific_block = np.tile(activation[id_number:id_number + uniq_stim,
                                             :], 
                                 (trials_per_block // 8, 1))
            
            # block them together
            shown_patterns[i,0:(trials_per_block//2),:]  = recur_block
            shown_patterns[i,(trials_per_block//2):,:]   = specific_block
            
            associated_resp[i,:,:] = answers
        
        # reshape to 1088 rows (trials) and 72 columns (identifying the pattern)
        stimuli = np.reshape(shown_patterns, (trials_per_block * (no_diff_map-1), input_units))
        
        # reshape the objective array made earlier
        objective = np.reshape(associated_resp, (trials_per_block * (no_diff_map-1), no_outputs))  
        
        # create trial identifiers to ease later data analysis
        trial_id       = np.zeros((trial_total, 3))
        counting       = np.zeros(input_units)
        returning_cols = np.arange(returning*uniq_stim, returning*uniq_stim + uniq_stim)
        
        for r in range(trial_total):
            stimulus      = np.nonzero(stimuli[r,:])[0][0]   
            trial_id[r,0] = stimulus                  
            
            if stimulus in returning_cols:
                trial_id[r,2] = 1
            else:
                pass
            
            counting[stimulus] += 1
            trial_id[r,1]       = counting[stimulus]
        
        return stimuli, trial_id, objective



#%%

# ---------------- #
# testing the code #
# dataHandler      #
# ---------------- #  

# =============================================================================
# path_1 = r'C:\Users\pieter\Downloads\GitHub\Pieter_H\PhD\Year 1\Modelling' 
# path_2 = r'\Huycke\Ruge paradigm\Code\Programmed experiment - extensive training\data'
# rel_path = path_1 + path_2
# randomHelper.dataHandler(path = rel_path, subs_id = 26)
# =============================================================================
    
# ---------------- #
# testing the code #
# latinSquare     #
# ---------------- #

# =============================================================================
# latin_numpy = randomHelper.latinSquare(subjects = 30, conditions = 5)
# print(latin_numpy)
            
# latin_numpy = randomHelper.latinSquare(to_balance=np.arange(1,6))
# print(latin_numpy)
# 
# stim, obj, trial = randomHelper.create_stim_order(4, 6, 6)
# print(stim)
# =============================================================================

# ---------------- #
# testing the code #
# stimulusOrder    #
# ---------------- #  

# =============================================================================
# latin_square_row = np.array([2., 3., 4., 5., 6.])
# stim, obj, trial = randomHelper.stimulusOrder(uniq_stim   = 4,
#                                               no_diff_map = 6,
#                                               returning   = 1, 
#                                               block_order = latin_square_row)
# =============================================================================
