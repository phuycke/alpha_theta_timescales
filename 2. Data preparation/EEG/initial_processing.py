#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%%

# subjects that are analyzed
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

# loop over the subjects
for SUB_ID in full_epochs:
        
    ICA        = True           # compute ICA
    PLOTTING   = False          # show diagnostic plots
    VERBOSE    = False          # let the functions print extra info?
    
    #%%
    
    # ---------------- #
    # PATH DEFINITIONS #
    # ---------------- #
    
    # the home directory where two subfolders are located: code
    general  = r'C:\Users\pieter\OneDrive - UGent\Projects\2019'
    specific = r'\overtraining - PILOT 3'
    path     = general + specific
    
    # GENERAL: these directories should already exist
    DATA     = path + r'\code\Analysis\Behavioral\General\Data'
    EEG_DIR  = path + r'\sourcedata\eeg\data'
    NPY_DIR  = path + r'\sourcedata\beh\experiment'
    CODE_DIR = path + r'\code\Analysis\EEG\Python'
    ROOT     = path + r'\{}\ses-01\eeg\Python'.format(SUB_ID)
    LOG_DIR  = ROOT
    
    # delete unneeded variables
    del(general, specific)
    
    #%%
    
    # -------------- #
    # IMPORT MODULES #
    # -------------- #
    
    import logging 
    import matplotlib.pyplot as plt
    import mne
    import numpy             as np
    import os
    import pandas            as pd
    import scipy.io
    
    os.chdir(CODE_DIR)
    
    from eeg_helper import preprocessing as preproc
    
    #%%
    
    # ------------- #
    # START LOGGING #
    # ------------- #
    
    log = os.path.join(LOG_DIR, 'stimulus_{}.log'.format(SUB_ID))
    
    # check if log file already exist, if so: delete
    if os.path.isfile(log):
        os.remove(log)
        
    # define the format and level of the logging (info and up is written)
    logging.basicConfig(filename = log,
                        level    = logging.INFO,
                        format   = '%(asctime)s - %(levelname)-8s - %(message)s', 
                        datefmt  = '%d/%m/%Y %H:%M:%S')
    logging.info('Analysis for {} started'.format(SUB_ID))

    #%%
    
    # ---------------------- #
    # BUILD FOLDER STRUCTURE #
    # ---------------------- #
    
    # checking our definitions
    if os.path.isdir(ROOT):
        logging.info('{} successfully accessed'.format(ROOT))
    else:
        logging.critical('{} > this path did not exist'.format(ROOT))
        os.makedirs(ROOT)
        logging.critical('{} > this path was created'.format(ROOT))
        
    # other folders that are needed to store the results in an orderly fashion
    RES_DIRS = ['events', 'man_clean', 'epoch', 'ica', 'figures']
    
    for folder in RES_DIRS:
        if not os.path.exists(os.path.join(ROOT, folder)):
            os.makedirs(os.path.join(ROOT, folder))
            logging.info("Creating %-9s folder" % (folder))
        else:
            logging.info("Folder named %-9s already exists" % (folder))
        
    del(path, folder)
    
    #%%
    
    # ------------------ #
    # DATA CHECK         #
    # AMOUNT OF SUBJECTS #
    # ------------------ #
    
    # check how many subjects we have
    num_subjects = len(os.listdir(EEG_DIR))
    
    if num_subjects == 0:
        logging.ERROR('Error at line 110 >>> Program execution halted.')
        raise FileNotFoundError('\nNo data available...\nMaybe something went wrong?')
    
    del num_subjects
    
    #%%
    
    # -------------------------- #
    # ADD EXPERIMENT INFORMATION #
    # -------------------------- #
    
    print("\n{} - PIPELINE STARTED".format(SUB_ID))
    
    """
    A note on storing information in the raw.info object
    
    mind that the parameters that can be stored in these variables is restricted
    specifically, the data formats are heavily restricted:
        sex should be an integer
        birth date should be in a Julian format
    This all has to do with the write functions that are used when saving objects
        Epochs.save() will, for example, use writeint( ) to write the sex of the 
            subjects to the Epoch.info object
    """
    
    # extra information on the experiment
    experiment_explained = 'Extensive training (pilot) - {}'.format(SUB_ID)
    experimenters        = 'Pieter Huycke + Pieter Verbeke'
    
    subject_info         = pd.read_pickle(os.path.join(DATA, "descriptives.pkl"))
    subject_sex          = subject_info['Sex'][int(SUB_ID[-2:]) - 1]
    if subject_sex == 'M':
        subject_sex = 1
    elif subject_sex == 'F':
        subject_sex = 2
    else:
        raise Exception("Sex is undefined")
    
    subject_hand = subject_info['Handedness'][int(SUB_ID[-2:]) - 1]
    if subject_hand == 'R':
        subject_hand = 1
    else:
        subject_hand = 2
        
    # save the information about the subject
    subject_information  = dict(his_id = SUB_ID,
                                sex    = subject_sex, 
                                hand   = subject_hand)
    
    # add this information to the object
    raw = preproc.add_eeg_info(eeg_file        = os.path.join(EEG_DIR, "{}.bdf".format(SUB_ID)),
                               loc             = EEG_DIR,
                               experiment_info = experiment_explained,
                               experimenters   = experimenters,
                               subject_info    = subject_information,
                               log_object      = logging)
    
    del experiment_explained, experimenters, subject_information, subject_sex, subject_hand, subject_info
    
    # search for events
    logging.info('Looking up trigger information...')
    
    trig_ms  = 0.03
    logging.info('Minimum trigger sending time: {0:.2f} seconds'.format(trig_ms))
    
    events   = mne.find_events(raw, 
                               stim_channel = 'STI 014', 
                               verbose      = VERBOSE, 
                               min_duration = trig_ms)
    
    del trig_ms
    
    event_id = dict(inspect_stim_start  = 40,
                    inspect_stim_end    = 41,
                    trial_start         = 50,
                    stim_repeat         = 60,
                    stim_non_repeat     = 61,
                    resp_correct        = 70,
                    resp_incorrect      = 71,
                    resp_too_late       = 72,
                    feedback_correct    = 80,
                    feedback_incorrect  = 81,
                    feedback_too_late   = 82,
                    trial_end_fixation  = 85,
                    block_end_message   = 100,
                    block_end_actual    = 101,
                    end_of_experiment   = 2)
    
    # write to dedicated folder (use absolute path instead of relative ones)
    logging.critical('Writing trigger info to dedicated folder...')
    np.save(os.path.join(ROOT, 'events', '{}_events.npy'.format(SUB_ID)), 
            events)
    
    #%%
    
    # -------------------------- #
    # ADD BEHAVIORAL INFORMATION #
    # -------------------------- #
    
    raw = preproc.add_beh_info(beh_file        = os.path.join(NPY_DIR, "{}.npy".format(SUB_ID)),
                               loc             = NPY_DIR,
                               eeg_file        = raw,
                               log_object      = logging)
    
    #%%
    
    # --------------------- #
    # SET ELECTRODE MONTAGE # 
    # --------------------- #
    
    montage = mne.channels.make_standard_montage('biosemi64')
    logging.critical('Applying biosemi 64 montage...')
    raw.set_montage(montage)
    
    #%%
    
    # ----------- #
    # REREFERENCE #
    # ----------- #
    
    # rereference to the average of the first and the second external electrode
    logging.critical('Rereference to the average mastoids...')
    raw.load_data()                              # load data in memory
    raw.set_eeg_reference(['EXG1', 'EXG2'])      # set reference
     
    #%%
    
    # ----------- #
    # INTERPOLATE #
    # ----------- #
    
    # do this after ICA
    # do ICA with the bad channels excluded
        # go for specific values, or a reduced number of components
    # do the inverse ICA
        # go from ICA space to the input space again
    # interpolate the data then
    
    eeg_cleaning   = pd.read_pickle(os.path.join(DATA, "eeg_cleaning_params.pkl"))
    indx           = int(SUB_ID[-2:]) - 1
    to_interpolate = eeg_cleaning['Bad channels'][indx]
    
    if type(to_interpolate) == list:
        raw.info['bads'] = to_interpolate
        raw.interpolate_bads()
        logging.critical('Interpolating bad channels...')
    elif np.isnan(to_interpolate):
        logging.critical('No channels to interpolate')
    else:
        raise Warning('Type of "to_interpolate" not understood')
    
    # remove unneeded channels
    raw.info['bads'] = ['EXG1', 'EXG2',      # reference channels
                        'EXG7', 'EXG8',      # not used in recording
                        'STI 014']           # trigger channel
    
    #%%
    
    # --------- #
    # FILTERING #
    # --------- #
    
    logging.critical('High pass filter the Raw data at 0.1 Hz...')
        
    # filter the epochs using a highpass filter: .1 Hz
    raw_f = raw.filter(l_freq     = .1,
                       h_freq     = None, 
                       fir_design = 'firwin',
                       verbose    = VERBOSE,
                       n_jobs     = 4)
    
    #%%
    
    # ------------------- #
    # EVENT-LOCKED EPOCHS #
    # ------------------- #

    logging.critical('STIMULUS-LOCKED: epoching the Raw (0.1 Hz) data...')

    # define the stimulus-locked epochs in the original data
    origin = mne.Epochs(raw_f, 
                        events, 
                        event_id = [event_id.get('resp_correct'), 
                                    event_id.get('resp_incorrect')], 
                        tmin      = -1, 
                        tmax      =  1,
                        proj      = False,
                        verbose   = VERBOSE,
                        preload   = True)
    
    logging.critical('STIMULUS-LOCKED: no baseline period defined')

    #%%
    
    # run this code if the man_clean folder is empty
    if not os.listdir(os.path.join(ROOT, 'man_clean')):
        
        # copy the origin epochs, and manually clean the data
        uncleaned = origin.copy()
        origin.plot(picks      = 'eeg',
                    scalings   = dict(eeg = 120e-6),
                    n_epochs   = 5,
                    n_channels = 64)
        
        # --------------------- #
        # UPDATE EPOCH METADATA #
        # --------------------- #
        
        # define the mapping from info object to metadata pandas df
        column_names = {'beh_stim_ID'          : 'Stimulus ID',
                        'beh_repetitions_gen'  : 'Overall repetitions',
                        'beh_repetitions_spec' : 'In block repetitions',
                        'beh_block_ID'         : 'Block ID',
                        'beh_block_trial_num'  : 'Trial number block',
                        'beh_answered'         : 'Response',
                        'beh_correct'          : 'Correct',
                        'beh_RT'               : 'Reaction times',
                        'beh_block_num_gen'    : 'Block number general', 
                        'beh_block_num_spec'   : 'Block number specific',
                        'beh_trial_num'        : 'Trial number overall'} 
        
        # define the order of the pandas df columns
        column_order = ['Trial number overall', 
                        'Trial number block',
                        'Block number general', 
                        'Block ID', 
                        'Block number specific', 
                        'Overall repetitions', 
                        'In block repetitions', 
                        'Stimulus ID', 
                        'Response', 
                        'Correct', 
                        'Reaction times']
        
        # update the actual metadata
        epochs, saved_epochs = preproc.update_metadata(original_epochs    = uncleaned, 
                                                       rejected_epochs    = origin,
                                                       metadata_col_names = column_names,
                                                       metadata_col_order = column_order,
                                                       log_object         = logging,
                                                       rejection_log      = None)
        
        # add nchan in info if missing
        try:
            print(epochs.info['nchan'])
        except KeyError:
            print('\nAdding nchan in info structure')
            epochs.info['nchan'] = len(epochs.info['ch_names'])
        
        # save the resulting data
        epochs.save(os.path.join(ROOT, 'man_clean', 'resp_{}_man_clean_epo.fif'.format(SUB_ID)), 
                    fmt       = 'single',   # recommended
                    overwrite = True,       # ignore warnings if file already exists
                    verbose   = VERBOSE)
        np.save(os.path.join(ROOT, 'man_clean', 'resp_{}_rejec_log.npy'.format(SUB_ID)),
                saved_epochs)    
        
        del column_names, column_order
        
    else:
        # set the working directory
        os.chdir(os.path.join(ROOT, 'man_clean'))
        
        # load in the data
        epochs       = mne.read_epochs('resp_{}_man_clean_epo.fif'.format(SUB_ID))
        saved_epochs = np.load('resp_{}_rejec_log.npy'.format(SUB_ID))
    
    #%%
    
    # --------- #
    # FILTERING #
    # --------- #
    
    logging.critical('Bandpass filtering the Raw data between 1 Hz and 40 Hz')
    
    # filter the epochs using a highpass filter: .1 Hz
    raw_hf = raw.filter(l_freq     = 1,
                        h_freq     = 40, 
                        fir_design = 'firwin',
                        verbose    = VERBOSE,
                        n_jobs     = 4)
    
    #%%
    
    # ------------------- #
    # EVENT-LOCKED EPOCHS #
    # ------------------- #
    
    logging.critical('STIMULUS-LOCKED: epoching the Raw (1 Hz - 40 Hz) data...')

    # define the stimulus-locked epochs in the original data
    epochs_hf = mne.Epochs(raw_hf, 
                           events, 
                           event_id = [event_id.get('resp_correct'), 
                                       event_id.get('resp_incorrect')], 
                           tmin      = -1, 
                           tmax      =  1,
                           proj      = False,
                           verbose   = VERBOSE,
                           preload   = True)
       
    # delete unneeded epochs
    dat             = epochs_hf.get_data()
    dat_reduc       = dat[saved_epochs, :, :]
    epochs_hf._data = dat_reduc
    
    events           = epochs_hf.events
    events_reduc     = events[saved_epochs, :]
    epochs_hf.events = events_reduc
    
    logging.critical('Original Raw Instance deleted to prevent shallow copying.')
    
    #%%
    
    # cd to the correct directory
    if os.path.isdir(os.path.join(ROOT, 'man_clean', 'MATLAB')):
        logging.info('man_clean\MATLAB successfully accessed')
    else:
        os.makedirs(os.path.join(ROOT, 'man_clean', 'MATLAB'))
        logging.info('man_clean\MATLAB successfully created')
    os.chdir(os.path.join(ROOT, 'man_clean', 'MATLAB'))
    
    # save data as a .mat file
    if '{}_orig.mat'.format(SUB_ID) not in os.listdir('.'):
        # swap the axes to match with the EEGLAB data format
        temp    = np.swapaxes(dat_reduc, 0, 1)           
        dat_mat = np.swapaxes(temp, 1, 2)
        # save as SUB_ID_orig.mat
        scipy.io.savemat('resp_{}_orig.mat'.format(SUB_ID), 
                         dict(x = dat_mat))
        del dat, dat_reduc, events, events_reduc, temp, dat_mat
    else:
        del dat, dat_reduc, events, events_reduc 

    #%%
    
    # ------------- #
    # CHECK FIG DIR #
    # ------------- #
    
    fig_dir = os.path.join(ROOT, "Figures")
    if len(os.listdir(fig_dir)) == 0:
        logging.info("Figures folder empty")
    else:
        logging.critical("Figures folder contains files")

    #%%
    
    # show me what you got
    fig_args = os.path.join(ROOT, "Figures", "Remaining stimulus-locked epochs.png")
    _        = epochs_hf.average().plot_topo(title            = "Remaining stimulus-locked epochs across channels",
                                             color            = 'black',
                                             background_color = 'w', 
                                             legend           = False, 
                                             show             = PLOTTING)
    _.savefig(fig_args)
    if not PLOTTING:
        plt.close('all')
        
    #%%
    
    # ------------ #
    # ICA          #
    # FITTING: MNE #
    # ------------ #
    
    # define the random seed and the removed components (after visual check)
    random_seed = int(eeg_cleaning['ICA seed used'][indx])
    removed_ICA = eeg_cleaning['Removed ICA components'][indx]
    
    # how to call the component plots
    figure_names = ['ICA components 01-20.png',
                    'ICA components 20-40.png'] 
    
    if ICA:
        from mne import preprocessing as preprocessing
        
        ica = preprocessing.ICA(method       = "infomax",
                                fit_params   = dict(extended = True), 
                                #random_state = random_seed,
                                n_components = 32,
                                verbose      = VERBOSE)
        
        # log start of ICA
        logging.critical('ICA (extended infomax): STARTED')
        logging.info('ICA (extended infomax): random seed = {}'.format(random_seed))
    	
        # fitting ICA takes somewhere between 600 and 700 seconds.
        ica.fit(epochs_hf,
                picks   = 'eeg', 
                verbose = VERBOSE) 

        # log start of ICA
        logging.critical('ICA (extended infomax): ENDED')
    
        # plot components: interactive mode
        counter = 0
        for components in ica.plot_components(ch_type = 'eeg',
                                              inst    = epochs_hf,
                            				    show    = PLOTTING):
            fig_args = os.path.join(ROOT, "Figures", figure_names[counter])
            components.savefig(fig_args)
            if not PLOTTING:
                plt.close('all')
            counter += 1
        
        # write to dedicated folder
        logging.critical('Writing ICA data to dedicated folder...')
        _ = ica.save(os.path.join(ROOT, 'ica', '{}-stim_ica.fif'.format(SUB_ID)))
        
        # optional: plot the time course of the components
        ica.plot_sources(inst = epochs_hf)
        
        # apply the blinking correction to the data: wb = without blinking
        epochs_wb = epochs.copy()
        epochs_wb.load_data()
        
        # remove the first component (sub-04 (eye blinking))
        if not np.isnan(removed_ICA).all():
            for i in range(len(removed_ICA)):
                logging.critical('Exluding the following component: {}'.format(removed_ICA[i]))
            ica.exclude = removed_ICA
           
        # apply the ica
        ica.apply(epochs_wb)
        
        # show me what you got
        fig_args = os.path.join(ROOT, "Figures", "Remaining stim-locked epochs after ICA.png")
        _        = epochs_wb.average().plot_topo(title            = "Stimulus locked epochs after ICA",
                                                 color            = 'black',
                                                 background_color = 'w', 
                                                 legend           = False, 
                                                 show             = PLOTTING)
        _.savefig(fig_args)
        if not PLOTTING:
            plt.close('all')
    else:
        epochs_wb = epochs.copy()
        
    #%%
    
    # ------------------------ #
    # SAVE CLEAN DATA AND EXIT #
    # ------------------------ #
    
    # save the clean stimulus locked epochs
    logging.critical('Writing fully cleaned Epoch data to dedicated folder...')
    epochs_wb.save(os.path.join(ROOT, 'epoch', '{}_stimulus_epo.fif'.format(SUB_ID)), 
                   overwrite = True, 
                   verbose   = VERBOSE)

  
    # final log
    logging.critical('FINAL LOG - preprocessing for {} completed without errors'.format(SUB_ID))
    logging.shutdown()
    
    del log
