#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

# ------- #
# IMPORTS #
# ------- #

import mne
import numpy  as np
import os
import pandas as pd

#%%

# --------- #
# FUNCTIONS #
# --------- #

#%%
   
class preprocessing:
    
    '''
    A class containing functions to aid preprocessing:
        - update the info object in the original RawArray object
        - update the info object by adding behavioral information
            - Reaction times, accuracies etc. can all be added
        - convert MATLAB time point to Python readable times
        - update the behavioral data by removing data associated with deleted 
          Epochs
    '''
      
    def add_eeg_info(eeg_file        = None,
                     loc             = None,
                     experiment_info = None,
                     experimenters   = None,
                     subject_info    = None,
                     log_object      = None):
        
                
        '''
        Update the information stored in the eeg data object (accessed via raw.info)
    
        :param str OR mne.io.array.array.RawArray eegfile: The eeg data file that is edited
        :param str loc: The location of the eeg data file
        :param str experiment_info: Some information on your experiment
        :param str experimenters: Who conducted the experiment?
        :param dict subject_info: Additional information you want to store
        :param module log_object: A reference to the log file used to store logs
        
        :type log_object: object or None
        :return: The raw data object with updated values
        :rtype: mne.io.array.array.RawArray
        
        :raises AttributeError: (not raised) if no log file is defined
        :raises Exception: if no eeg data file is defined
        :raises TypeError: if the eeg file is not of type str or RawArray
        :raises ValueError: if the eeg file is of type str but extension is not .bdf
        :raises FileNotFoundError: if loc refers to a non-existing path
        '''
        
        able_to_log   = False
        function_name = 'add_eeg_info'
        
        if log_object is not None:
            try:
                log_object.critical('{} - STARTED'.format(function_name))
                able_to_log = True
            except AttributeError:
                pass
        
        # if logging > disable verbose
        verbose_level = True
        if able_to_log:
            verbose_level = False
            
        if not able_to_log:
            print('')
            print(' * * * * * * * * * * * * * * * * * *')
            print(' * FUNCTION CALL >>> add_eeg_info  *')
            print(' * * * * * * * * * * * * * * * * * *')
            print('')
            
        # file name operations
        if (eeg_file is None):
            if able_to_log:
                log_object.error('error in {}: check 1'.format(function_name))
            raise Exception('\n!! Conflict in passed parameters !!\n'
                            'The filename of your EEG data should be passed')
        if (type(eeg_file) is not str) and (type(eeg_file) is not mne.io.array.array.RawArray):
            if able_to_log:
                log_object.error('error in {}: check 2'.format(function_name))
            raise TypeError('\nFilename should be either\n\ta string\n\ta RawArray\n')
        if (type(eeg_file) is str) and eeg_file[-4:].lower() != '.bdf':
            if able_to_log:
                log_object.error('error in {}: check 3'.format(function_name))
            raise ValueError('\nThis pipeline was written to handle .bdf files\n'
                             'Option 1:\n'
                             '\tAdjust the source code to read other file formats\n'
                             'Option 2:\n'
                             '\tConvert your EEG data file to the .bdf format')
        
        # declare
        raw = None
        
        # decision tree
        if type(eeg_file) is str:
            if loc is None:
                if able_to_log:
                    log_object.error('error in {}: check 4'.format(function_name))
                raise FileNotFoundError('\nNo path was provided.\nWe cannot find the data')
            else:
                if os.path.isdir(loc):
                    path_to_file = os.path.join(loc, eeg_file)
                    raw          = mne.io.read_raw_bdf(path_to_file, 
                                                       verbose = verbose_level)
                else:
                    if able_to_log:
                        log_object.error('error in {}: check 5'.format(function_name))
                    raise FileNotFoundError("The path is faulty, please provide the exact path to the EEG file...")
            
        else:
            raw = eeg_file
        
        info  = raw.info
        
        if able_to_log:
            log_object.info('{}: initial checks passed'.format(function_name))
        else:
            print('\n* LOG *\nInitial checks passed')
    
        # add string to the description tag
        if experiment_info is not None:
            info['description']     = experiment_info
        
        if experimenters is not None:
            info['experimenter']    = experimenters
        
        if subject_info is not None:
            info['subject_info']    = subject_info
                   
        # set the correct EEG channel info
        ch_names = ['Fp1', 'AF7', 'AF3', 'F1',  'F3',  \
                    'F5',  'F7',  'FT7', 'FC5', 'FC3', \
                    'FC1', 'C1',  'C3',  'C5',  'T7',  \
                    'TP7', 'CP5', 'CP3', 'CP1', 'P1',  \
                    'P3',  'P5',  'P7',  'P9',  'PO7', \
                    'PO3', 'O1',  'Iz',  'Oz',  'POz', \
                    'Pz',  'CPz', 'Fpz', 'Fp2', 'AF8', \
                    'AF4', 'AFz', 'Fz',  'F2',  'F4',  \
                    'F6',  'F8',  'FT8', 'FC6', 'FC4', \
                    'FC2', 'FCz', 'Cz',  'C2',  'C4',  \
                    'C6',  'T8',  'TP8', 'CP6', 'CP4', \
                    'CP2', 'P2',  'P4',  'P6',  'P8',  \
                    'P10', 'PO8', 'PO4', 'O2']
        
        if 'A1' in info['ch_names']:
            info['ch_names'][:len(ch_names)] = ch_names
            ch_names += info['ch_names'][len(ch_names):]
            for i in range(len(info['ch_names'])):
                info['chs'][i]['ch_name'] = ch_names[i]
        
        # add channel types (FPPW configuration)
        if type(eeg_file) is str:
            for i in range(73):
                if i < 64:
                    raw.set_channel_types(mapping={info['ch_names'][i]: 'eeg'})
                elif 64 <= i < 72:
                    raw.set_channel_types(mapping={info['ch_names'][i]: 'eog'})
                else:
                    raw.set_channel_types(mapping={info['ch_names'][i]: 'stim'})
            if able_to_log:
                log_object.info('{}: electrode types added'.format(function_name))
            else:
                print('\n* LOG *\nElectrode types added\n')
        
        # rename the trigger channel from Status to STI 014
        try:
            mne.rename_channels(info,
                                {'Status' : 'STI 014'})
        except ValueError:
            if able_to_log:
                log_object.error('Error in {}: check 6'.format(function_name))
            else:
                print('\nStatus could not be found in channel names.\nContinuing...\n')
        
        # return adjusted data
        if able_to_log:
            log_object.critical('{}: COMPLETED'.format(function_name))
        else:
            print("\n* LOG *\n{}\n\tadd_eeg_info() completed\n".format(eeg_file))
        return raw
    
    
    #%%
     
        
    def add_beh_info(beh_file        = None,
                     loc             = None, 
                     eeg_file        = None,
                     log_object      = None):
        
        '''
        Add behavioral data stored in a .npy file to the eeg data object
    
        :param str beh_file: The data file that is edited
        :param str loc: The location of the eeg data file
        :param mne.io.array.array.RawArray eegfile: The eeg data file read in mne
        :param module log_object: A reference to the log file used to store logs
        
        :type log_object: object or None
        :return: The raw eeg data object with added behavioral information
        :rtype: mne.io.array.array.RawArray
        
        :raises Exception: if no eeg data file and no behavioral file is defined
        :raises TypeError: if the eeg file is not of type str or RawArray, or
                           if behavioral file is not of type str
        :raises ValueError: if the beh file is of type str but extension is not .npy
                            if the eeg file is not of type RawEDF
        
        :raises FileNotFoundError: if loc refers to a non-existing path
        '''
        
        able_to_log   = False
        function_name = 'add_beh_info'
        
        if log_object is not None:
            try:
                log_object.critical('{} - STARTED'.format(function_name))
                able_to_log = True
            except AttributeError:
                pass
            
        if not able_to_log:
            print('')
            print(' * * * * * * * * * * * * * * * * * *')
            print(' * FUNCTION CALL >>> add_beh_info  *')
            print(' * * * * * * * * * * * * * * * * * *')
            print('')
        
        # file name operations
        if (beh_file is None) or (eeg_file is None):
            if able_to_log:
                log_object.error('error in {}: check 1'.format(function_name))
            raise Exception('\n!! Conflict in passed parameters !!\n'
                            'The filenames of your behavioral- and EEG data should be passed')
        if (type(beh_file) is not str):
            if able_to_log:
                log_object.error('error in {}: check 2'.format(function_name))
            raise TypeError('\nBehavioral file should be of type\n\tString')
        else:
            if beh_file[-4:].lower() != '.npy':
                if able_to_log:
                    log_object.error('error in {}: check 3'.format(function_name))
                raise TypeError('\nBehavioral file extension should be ".npy"')
            
        if (type(eeg_file) is not mne.io.edf.edf.RawEDF):
            if able_to_log:
                log_object.error('error in {}: check 4'.format(function_name))
            raise ValueError('\nThe EEG file should be of type:\n\tRawEDF')
        
        if loc is None:
            if able_to_log:
                log_object.error('error in {}: check 5'.format(function_name))
            raise FileNotFoundError('\nNo path was provided.\nWe cannot find the data')
        else:
            if os.path.isdir(loc):
                path_to_file = os.path.join(loc, beh_file)
                dat          = np.load(path_to_file)
            else:
                if able_to_log:
                    log_object.error('error in {}: check 6'.format(function_name))
                raise FileNotFoundError("The path is faulty, please provide the exact path to the EEG file...")
        info = eeg_file.info             # access the info object in Raw
        
        if able_to_log:
            log_object.info('{}: initial checks passed'.format(function_name))
        else:
            print('* LOG *\nInitial checks passed')
    
        # create variable for block number
        blocks_gen  = np.repeat(np.arange(1, (len(dat) // 32) + 1), 32)
        
        # create a variable for trial number
        trials      = np.arange(1, len(dat) + 1)
    
        # create a variable for specific block numbers (rep vs non rep)
        blocks_spec = np.repeat(np.repeat(np.arange(1, 9), 2), 32)
        block_trial = np.tile(np.arange(1, 33), (len(dat) // 32))

        # create a variable for in block repetition count
        rep_spec    = np.zeros(len(dat))
        
        # loop over trials to determine in block repetition count
        for indx in range(len(rep_spec)):
            # skip the cases where the stimulus is non repeating
            if dat[indx,2] == 0:
                rep_spec[indx] = dat[indx,1]
            else:
                if blocks_spec[indx] == 1:
                    rep_spec[indx] = dat[indx,1]
                else:
                    rep_spec[indx] = dat[indx,1] - ((blocks_spec[indx] - 1) * 8)               

        # assign info stored in behavioral file in the RawEDF info object
        info['beh_stim_ID']          = dat[:,0]
        info['beh_repetitions_gen']  = dat[:,1]
        info['beh_repetitions_spec'] = rep_spec
        info['beh_block_ID']         = dat[:,2]
        info['beh_block_trial_num']  = block_trial
        info['beh_answered']         = dat[:,4]
        info['beh_correct']          = dat[:,5]
        info['beh_RT']               = dat[:,6]
        info['beh_block_num_gen']    = blocks_gen
        info['beh_block_num_spec']   = blocks_spec
        info['beh_trial_num']        = trials
    
        # return adjusted data
        if able_to_log:
            log_object.critical('{} - COMPLETED'.format(function_name))
        else:
            print("\n* LOG *\n{}\n\tadd_beh_info() completed\n".format(beh_file))
        return eeg_file
    
    
    #%%
        
    
    def convert_samples_to_anot(eeghist       = None, 
                                sampling_freq = None,
                                log_object    = None):
        
        '''
        Convert MATLAB time point to our time points
    
        :param numpy.Array eeghist: An array containing MATLAB time points
        :param int sampling_freq: Sampling frequency used during EEG recording
        :param module log_object: A reference to the log file used to store logs
        
        :type log_object: object or None
        :return: Two arrays, one containing start points, the other lengths
        :rtype: numpy.Array
        
        :raises Exception: if no eeghist is defined
        :raises TypeError: if eeghist is not of type numpy.Array
        :raises ValueError: if the sampling frequency is not an int
        '''
        
        able_to_log   = False
        function_name = 'convert_samples_to_anot'
        
        if log_object is not None:
            try:
                log_object.critical('{} - STARTED'.format(function_name))
                able_to_log = True
            except AttributeError:
                pass
            
        if not able_to_log:
            print('')
            print(' * * * * * * * * * * * * * * * * * * * * * * *')
            print(' * FUNCTION CALL >>> convert_samples_to_anot *')
            print(' * * * * * * * * * * * * * * * * * * * * * * *')
            print('')
        
        # file name operations
        if (eeghist is None):
            if able_to_log:
                log_object.error('error in {}: check 1'.format(function_name))
            raise Exception('\n!! Conflict in passed parameters !!\n'
                            'A MATLAB history file should be passed')
            
        if type(eeghist) is not np.ndarray:
            if able_to_log:
                log_object.error('error in {}: check 2'.format(function_name))
            raise TypeError('\nThe inputted variable should be of the type numpy.ndarray\n')
            
        if type(sampling_freq) is not int:
            if able_to_log:
                log_object.error('error in {}: check 3'.format(function_name))
            raise ValueError('\nThe provided sampling frequency should be an integer.\n')
        
        if sampling_freq is None:
            if able_to_log:
                log_object.error('error in {}: check 4'.format(function_name))
            else:
                print('Warning\nNo sampling frequency provided.\nWe will assume a sampling rate of 1024 Hz.')
            sampling_freq = 1024
        
        # convert samples to time in seconds
        in_sec    = eeghist / sampling_freq
        
        # determine the onset valuesa
        onsets    = np.zeros(len(eeghist))
        durations = np.zeros(len(eeghist))
        
        for i in range(len(in_sec)):
            onsets[i]    = in_sec[i][0]
            durations[i] = in_sec[i][1] - in_sec[i][0]
        
        # log ebd of func
        if able_to_log:
            log_object.critical('{} - COMPLETED'.format(function_name))
        
        # return the results
        return onsets, durations
    
    
    #%%
        
    
    def update_metadata(original_epochs    = None,
                        rejected_epochs    = None,
                        metadata_col_names = dict,
                        metadata_col_order = None,
                        log_object         = None,
                        rejection_log      = None):
        
        '''
        Find deleted Epochs, and update the behavioral data accordingly
    
        :param mne.epochs.Epochs original_epochs: Object containing the data
         BEFORE any epoch rejection was done
        :param mne.epochs.Epochs rejected_epochs: Object containing the data
         AFTER epoch rejection was done
        :param dict metadata_col_names: A dictionary to map the info names to
         pandas df column names
        :param list metadata_col_order: (optional) A list to define the order 
         of the pandas df columns
        :param module log_object: A reference to the log file used to store logs
        
        :type log_object: object or None
        :return: rejected_epochs with info object and metadata object
        :rtype: mne.epochs.Epochs
        
        :raises TypeError: if first two params are not mne.epochs.Epochs
        :raises Exception: if the search procedure failed
        '''
        
        able_to_log   = False
        function_name = 'update_metadata'
        
        if log_object is not None:
            try:
                log_object.critical('{} - STARTED'.format(function_name))
                able_to_log = True
            except AttributeError:
                pass
            
        if not able_to_log:
            print('')
            print(' * * * * * * * * * * * * * * * * * * *')
            print(' * FUNCTION CALL >>> {} *'.format(function_name))
            print(' * * * * * * * * * * * * * * * * * * *')
            print('')
        
        if (type(original_epochs) is not mne.epochs.Epochs) and (type(rejected_epochs) is not mne.epochs.Epochs):
            if able_to_log:
                log_object.error('error in {}: check 1'.format(function_name))
            raise TypeError("The input arguments must be of type\n\tmne.epochs.Epochs")
        
        if type(metadata_col_names) is not dict:
            raise TypeError("'metadata_col_names' must be of type dict")
        
        if rejection_log is None:
            original = original_epochs.get_data()
            rejected = rejected_epochs.get_data()
            
            searching = True
            
            while searching:
                # create array with shape 73 x 512: 73 electrodes x 512 epochs
                temp    = np.zeros((original.shape[1], 
                                    original.shape[0]))
            
                # draw a random time point that we will use to compare epochs
                timepoint = np.random.randint(0, original.shape[2])
                
                # loop over epochs, and write all electrode values at the chosen timepoint
                for i in range(len(original)):
                    # for each epoch i, we store all electrode values at timepoint 'timepoint'
                    temp[:,i] = original[i,:,timepoint]
                
                # initialize    
                first_unique = None
                electrode    = None
                
                for j in range(temp.shape[0]):
                    if len(np.unique(temp[j,:])) == temp.shape[1]:
                        first_unique = temp[j,:]
                        electrode    = j
                        if able_to_log:
                            log_object.critical('{} - SEARCH IS A SUCCESS'.format(function_name))
                        else:
                            print('\nSuccess!\n')
                        break
                
                if first_unique is None:
                    if able_to_log:
                        log_object.error('{}: no unique row was found...'.format(function_name))
                    raise Exception('No unique row was found...')
                else:
                    if able_to_log:
                        log_object.critical('{0}: selected electrode = {1}'.format(function_name, j))
                        log_object.critical('{0}: selected timepoint = {1}'.format(function_name, timepoint))
                    else:
                        print('Selected electrode:\n\t{0}\nSelected timepoint:\n\t{1}'.format(j, timepoint))
                    electrode = j
                    searching = False
            
            # create array with shape 73 x 492: 73 electrodes x 492 epochs
            temp_epoched = np.zeros((1, 
                                     rejected.shape[0]))
            
            # select the electrode and timepoint combination we need
            temp_epoched = rejected[:,electrode,timepoint]
            counter      = 0
            indxs        = np.zeros(len(first_unique))
            
            rejec_log = [True] * len(original_epochs)
            for j in range(len(first_unique)):
                # the epochs that are found in both Epoch objects receive a 1
                if np.isin(first_unique[j], temp_epoched):
                    indxs[j] = 1
                else:
                    # epochs that are not shared between objects stay zero
                    if able_to_log:
                        log_object.info('{0}: epoch {1:3d} was dropped'.format(function_name, j + 1))
                        rejec_log[j] = False
                    else:
                        print("Epoch {0:3d} was dropped".format(j + 1))
                    # keep track of the amount of dropped epochs
                    counter += 1
            
            # keeping count
            if able_to_log:
                log_object.critical('{0}: {1} epoch(s) dropped'.format(function_name, counter))
            else:
                print("\n* LOG *\nA total of {} epochs dropped".format(counter))
            
            # only select the data points where the indx equals zero (see above)
            indx = np.ravel(np.where(indxs == 1)[0])
            
            # reject the actual behavioral data where the epochs were dropped
            for info_label in rejected_epochs.info:
                # only change the info if it is associated with behavioral data
                if 'beh' in info_label:
                    if len(original_epochs) != 512:
                        rejected_epochs.info[info_label] = rejected_epochs.info[info_label][-len(original_epochs):]
                    rejected_epochs.info[info_label] = rejected_epochs.info[info_label][indx]
            
            # add all the behavioral information to the metadata
            meta_df  = pd.DataFrame()
            colnames = metadata_col_names
            
            # loop over the info entries that contain behavioral data
            for label in rejected_epochs.info:
                if 'beh' in label:
                    meta_df[colnames[label]]    = rejected_epochs.info[label]
                    rejected_epochs.info[label] = None
            
            # loop over all info labels and delete beh ones
            while True:
                beh_found = False
                for label in rejected_epochs.info:
                    if 'beh' in label:
                        beh_found = True
                        break
                del rejected_epochs.info[label]
                if not beh_found:
                    break
                
            # reorder the columns of the metadata if this is deemed necessary
            if metadata_col_order is not None:
                if type(metadata_col_order) is not list:
                    raise TypeError("'metadata_col_order' must be of type list")
                meta_df = meta_df[metadata_col_order]
                
            rejected_epochs.metadata = meta_df
    
            # end message            
            if able_to_log:
                log_object.info('{}: metadata set + info entries removed'.format(function_name))
                log_object.critical('{} - COMPLETED'.format(function_name))
            else:
                print("\n* LOG *\nmetadata set + info entries removed\nReturning updated Epoch object...\n")
            
            return rejected_epochs, np.ravel(np.array(rejec_log))
        
        else:
        
            indx = rejection_log
            
            # add needed information
            rejected_epochs.info['nchan']    = len(rejected_epochs.info['ch_names'])
            rejected_epochs.info['ch_names'] = original_epochs.info['ch_names']
            
            if len(indx) == 512:
                # reject the actual behavioral data where the epochs were dropped
                for info_label in original_epochs.info:
                    # only change the info if it is associated with behavioral data
                    if 'beh' in info_label:
                        rejected_epochs.info[info_label] = original_epochs.info[info_label][indx]
            else:
                for info_label in original_epochs.info:
                    # only change the info if it is associated with behavioral data
                    if 'beh' in info_label:
                        original_epochs.info[info_label] = original_epochs.info[info_label][-len(indx):]
                        rejected_epochs.info[info_label] = original_epochs.info[info_label][indx]
            
            # add all the behavioral information to the metadata
            meta_df  = pd.DataFrame()
            colnames = metadata_col_names
                       
            # loop over the info entries that contain behavioral data
            for label in rejected_epochs.info:
                if 'beh' in label:
                    meta_df[colnames[label]]    = rejected_epochs.info[label]
                    rejected_epochs.info[label] = None
            
            # loop over all info labels and delete beh ones
            while True:
                beh_found = False
                for label in rejected_epochs.info:
                    if 'beh' in label:
                        beh_found = True
                        break
                del rejected_epochs.info[label]
                if not beh_found:
                    break
            
            # reorder the columns of the metadata if this is deemed necessary
            if metadata_col_order is not None:
                if type(metadata_col_order) is not list:
                    raise TypeError("'metadata_col_order' must be of type list")
                meta_df = meta_df[metadata_col_order]
                
            rejected_epochs.metadata = meta_df
    
            # keep track of deleted Epochs
            if able_to_log:
                log_object.critical('{0} - {1} Epochs removed'.format(function_name,
                                                                      (512 - len(meta_df))))
            deleted  = np.where(rejection_log == False)[0]
            deleted += 1
            
            for i in range(len(deleted)):
                if able_to_log:
                    log_object.info('{0}: epoch {1:3d} was dropped'.format(function_name,
                                                                           deleted[i]))
            
            # end message            
            if able_to_log:
                log_object.info('{}: metadata set + info entries removed'.format(function_name))
                log_object.critical('{} - COMPLETED'.format(function_name))
            else:
                print("\n* LOG *\nmetadata set + info entries removed\nReturning updated Epoch object...\n")
            
            return rejected_epochs
    
