#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Pieter Huycke
email:   pieter.huycke@ugent.be
GitHub:  phuycke
"""

#%% 

# ----------- #
# DEFINITIONS #
# ----------- #

# BOOLEANS 
EEG_measure           = False      # send triggers or not?
full_screen           = False      # use full screen or not?
speedy                = True      # speedy mode activated or not? 

# INT
subject_number        = 2          # the subject number

# FLOATS
max_resp_time         = 1.5        # answer within 1500 ms, or too late
speedy_wait           = .01        # stimulus time for speedy mode 
stimulus_presentation = 0.2        # stimulus shown for 200 ms

#%%

# -------------- #
# IMPORT MODULES #
# -------------- #

import logging
import os
import numpy as np

from psychopy            import clock, core, event, parallel, visual

# define the root folder according to the BIDS structure
root     = r'C:\Users\pieter\Downloads\GitHub\Pieter_H\2018-2019\projects\ruge\extensive training - experiment\overtraining'

# cd to correct dir to import randomizationHelper
code_dir = root + r'\code\experiment'
os.chdir(code_dir)

from randomizationHelper import randomHelper as helper

#%%

np.set_printoptions(suppress  = True)
np.set_printoptions(linewidth = 200)

# ---------------- #
# HELPER FUNCTIONS #
# ---------------- #

def folder_search(path = None, 
                  make_dirs = False, 
                  cd_to = None):
    
    """
    :param path:
    The path to the directory relevant for our purposes
    This path can lead to the data directory, stimulus folder, gaol folder...
    String (raw)

    :param make_dirs:
    Command used to create an 'experiment' and 'exercise' folder in the folder
    specificied by the path parameter
    Boolean
         
    :param cd_to:
    Simply switch the current directory to the directory specified in path
    Boolean
    
    :return:
    make_dirs and cd_to cannot be True at the same time
    Depends on the use of the two aforementioned parameters
    
    :example use:
    folder_search(path       = directory/where/data/is/stored)
    
    Will swith from working directory to path
    """
    
    if path is None:
        raise FileNotFoundError('Please define the path to the folder where '
                                'the data should be stored.')
    else:
        try:
            os.chdir(path)
        except FileExistsError or FileNotFoundError:
            print('The path you specified does not exist.\n'
                  'Please review the inputted path.\n')
    if make_dirs and cd_to is not None:
        raise Exception('You cannot make folders and cd to these folders at the same time.')
    else:
        if make_dirs and cd_to is None:
            folders = ['exercise', 'experiment']
            for dirs in folders:
                rel_path = path + '\\' + dirs
                if not os.path.exists(rel_path):
                    os.mkdir(rel_path)
                    os.chdir(rel_path)
                    os.mkdir('temp')
                else:
                    pass
                
        elif not make_dirs and cd_to is not None:
            os.chdir(path)
            try:
                os.chdir(cd_to)
            except FileExistsError or FileNotFoundError:
                print('The path you specified does not exist.\n'
                      'Please review the inputted path.\n')


def make_exercise_block(num_exercise_blocks = None,
                        sub_num             = None):
    
    """
    :param num_exercise_blocks:
    The amount of exercise blocks in this experiment
    Integer

    :param sub_num:
    The subject number
    Integer
         
    :return:
    Returns an exercise block for this specific subject
    
    :example use:
    make_exercise_block(num_exercise_blocks = 2,
                        sub_num             = 1)
    
    Returns 64 exercise trials (32 trials per specific mapping) for subject 1
    """
    
    if subject_number is None:
        raise ValueError('A subject number should be specified')
    
    # determine the mapping order
    recurrent_mapping = np.array(1)
    block_order       = np.arange(num_exercise_blocks, num_exercise_blocks+1)
    actual_order      = np.zeros(num_exercise_blocks)
    
    for i in range(0, num_exercise_blocks, 2):
        actual_order[i]   = recurrent_mapping
        actual_order[i+1] = block_order[i // 2]
    
    # store the data for a specific subject
    trials   = np.zeros((num_exercise_blocks, 32, 1))
    trial_id = np.zeros((num_exercise_blocks, 32, 3))
    response = np.zeros((num_exercise_blocks, 32, 1))
    
    # loop over stimulus codes and create the needed arrays using the function
    for i in range(len(actual_order)):
        stimuli_used = np.arange((actual_order[i] - 1) * 4,
                                  actual_order[i] * 4)
        
        returning_mapping = False
        if actual_order[i] == recurrent_mapping:
            returning_mapping = True
        
        trial_arr, trial_ident, response_array = helper.probabilistic_shuffle(stimulus_array=stimuli_used, 
                                                                              trials=32,
                                                                              returning=returning_mapping)
        
        # reshape arrays in correct format and store away
        trials[i,:,:]   = np.reshape(trial_arr, trials[i,:,:].shape)
        trial_id[i,:,:] = np.reshape(trial_ident, trial_id[i,:,:].shape)
        response[i,:,:] = np.reshape(response_array, response[i,:,:].shape)
        
    # reshape to create 320 trials
    tr     = np.reshape(trials,   (32 * num_exercise_blocks, 1))
    tr_id  = np.reshape(trial_id, (32 * num_exercise_blocks, 3))
    resp   = np.reshape(response, (32 * num_exercise_blocks, 1))
    
    # add an arbitrary value to make sure that the exercise stimuli are different from experimental ones
    random_number = np.random.randint(low = 50, high = 100)
    tr_id[:,0]   += np.array(random_number)
    
    return tr, tr_id, resp


def send_triggers(port_address   = 0xC020, 
                    trigger_time = .03,  
                    init         = False, 
                    sending      = True, 
                    value        = None):
    
    """
    :param port_address:
    The address of the parallel port
    Hex

    :param trigger_time:
    How long a trigger has to be sent, keeping in mind the sampling rate
    Integer
         
    :param init:
    Initialize the EEG triggers if set to True
    Boolean
    
    :param sending:
    Send a trigger if set to True, will send the integer defined in 
    the parameter value
    Boolean
    
    :param value:
    The trigger value sent
    if value is set to '255', then [255 0] will appear when looking at the 
    ActiChamp window when triggers are sent
    Integer
    
    :return:
    Depends on the use:
        - Initialize triggers
        - Send a specific integer as trigger value
    
    :example use:
    send_triggers(init    = True,   # initialize triggers
                  sending = False)
    send_triggers(value   = 160)    # sends trigger with value '160'
        
    Note that an error will occur if both sending and init are set to True
    """
    
    try:
        if sending and init:
            raise ValueError('You cannot both initialize the triggers, and send at the same time')
            return 1                                  # fail
        elif not sending and not init:
            raise ValueError('You did not specify what you want to do with this function')
            return 1                                  # fail
        else:
            if init:
                parallel.setPortAddress(port_address) # address of the parallel port in the Biosemi setup
                parallel.setData(0)                   # set all pins to low
                core.wait(trigger_time)               # wait for x seconds
            else:
                if value is None:
                    raise Exception('No value for trigger defined')
                    return 1                          # fail
                if not isinstance(value, int):
                    raise Exception('Trigger value should be an integer')
                    return 1                          # fail
                
                parallel.setData(value)               # sets all pins high
                core.wait(trigger_time)               # wait for x seconds
                parallel.setData(0)                   # sets all pins low
            return 0                                  # success!
    except RuntimeError or NameError:
        print('\nLOG: sending of EEG trigger failed\nAn error occured')
        return 2                                      # exception


#%%

# ------------------ #
# FOLDER DEFINITIONS #
# ------------------ #

# general definitions
data_dir = root + r'\sub-{:02d}'.format(subject_number)
stim_dir = root + r'\stimuli'

# go to the correct directory based on subject number
helper.dataHandler(path    = root, 
                   subs_id = subject_number)

# create the desired folder structure 
subject_dir        = os.getcwd()
folder_search(path = subject_dir, make_dirs = True)
os.chdir(subject_dir)

#%%

# ------------- #
# START LOGGING #
# ------------- #

if not os.path.exists('log'):
    os.mkdir('log')
log = "log\sub{:02d}.log".format(subject_number)
logging.basicConfig(filename = log,
                    level    = logging.DEBUG,
                    format   = '%(asctime)s.%(msecs)03d %(message)s', 
                    datefmt  = '%d/%m/%Y %H:%M:%S')
logging.info('INFO:\tSubject {:02d}: experiment started'.format(subject_number))

#%%

# ------------------------------ #
# LOAD DATA + MAKE NEW VARIABLES #
# ------------------------------ #

'''
Notes on the encoding
    Respones:
        - 0 equals LEFT
        - 1 equals RIGHT
    Fixation type:
        - 0 equals black  for focus, blue   for relax
        - 1 equals blue   for focus, black  for relax
'''

# load predetermined data
random_param = np.load('sub-{:02d}-rand param.npy'.format(subject_number))
trials       = np.load('sub-{:02d}-trial.npy'.format(subject_number))
trial_id     = np.load('sub-{:02d}-trial id.npy'.format(subject_number))
expected     = np.load('sub-{:02d}-response.npy'.format(subject_number)) 
    
# determine the encoding for the fixation crosses
rel_font = foc_font = 'wolowolo'

if int(random_param[subject_number - 1][1]) == 0:
    foc_font = 'black'
    rel_font = 'dodgerblue'
else:
    foc_font = 'dodgerblue'
    rel_font = 'black'   

if rel_font == 'wolowolo' or foc_font == 'wolowolo':
    raise Exception('\nLOG: color naming went wrong, check line 321 - 326')

# amount of blocks
no_exercise_blocks = 1
no_blocks          = len(trials) // 64
num_blocks         = no_exercise_blocks + no_blocks

# stimulus mapping for actual experiment
stimulus_mapping = np.arange(0, int(max(trial_id[:,0])) + 1)
figures          = np.add(stimulus_mapping, 1)
np.random.shuffle(figures)

logging.info('INFO:\tSubject {:02d}: experiment started'.format(subject_number))
logging.info('INFO:\tStimulus presentation time: {0:.2f} milliseconds'.format(stimulus_presentation * 1000))
logging.info('INFO:\tResponse window: {0:.2f} milliseconds'.format(max_resp_time * 1000))
logging.info('INFO:\tAmount of exercise blocks: {0:02d}'.format(no_exercise_blocks) * 2)
logging.info('INFO:\tAmount of experiment blocks: {0:02d}'.format(no_blocks * 2))
logging.info('INFO:\tRecurrent mapping: {0:02d}'.format(int(random_param[subject_number - 1][0])))
logging.info('INFO:\tFixation type: FOCUS: {0:s} / RELAX: {1:s}'.format(foc_font, rel_font))

#%%

# ------ #
# JITTER #
# ------ #

# drawn from a uniform distribution (lower bound = 200, upper bound = 530)
jitter = np.zeros((num_blocks * 64, 4))

# create distributions for fixation cross and anticipation
for row in range(len(jitter)):
    drawn         = np.zeros(4)
    drawn[0]      = np.ravel(np.random.uniform(low = 1000, high = 2000))
    drawn[3]      = 3000 - drawn[0]
    drawn[1]      = np.ravel(np.random.uniform(low = 500, high = 750))
    drawn[2]      = 2000 - drawn[1]
    jitter[row,:] = drawn

jitter = np.split(jitter, 
                  num_blocks * 2)

#%%

# --------------- #
# EXPERIMENT DATA #
# --------------- #

# exercise block
exc_trials, exc_id, exc_resp = make_exercise_block(num_exercise_blocks = 2, 
                                                   sub_num             = subject_number)

# reshape exc data to match dimensions with the exp data
exc_trials = np.reshape(exc_trials, (1, 64, 1))
exc_id     = np.reshape(exc_id,     (1, 64, 3))
exc_resp   = np.reshape(exc_resp,   (1, 64, 1))

# experiment block
exp_trials = np.reshape(trials,   (len(trials)   // 64, 64, 1))
exp_id     = np.reshape(trial_id, (len(trial_id) // 64, 64, 3))
exp_resp   = np.reshape(expected, (len(expected) // 64, 64, 1))

# stack all trial info in one array to read during experiment
trial      = np.concatenate((exc_trials, exp_trials))
trial_id   = np.concatenate((exc_id,     exp_id))
trial_resp = np.concatenate((exc_resp,   exp_resp))
    
logging.info('INFO:\tData successfully reshaped in {0:02d} blocks'.format(num_blocks))

#%%

# --------------------- #
# RANDOMIZE BLOCK ORDER #
# --------------------- #

for block in range(num_blocks):
    # split the blocks in two
    considered_trial = np.split(trial[block], 2) 
    considered_id    = np.split(trial_id[block], 2)
    considered_resp  = np.split(trial_resp[block], 2)
    
    if (len(considered_trial[0]) == len(considered_trial[1]) == 32) and \
       (len(considered_id[0]) == len(considered_id[1]) == 32) and \
       (len(considered_resp[0]) == len(considered_resp[1]) == 32):
           logging.info('INFO:\tDimensions for random block shuffling check out')
           logging.info('INFO:\t\tMessage for loop {0:d}/{1:d}'.format(block+1, num_blocks))

# split experiment in blocks of 32 trials
trial      = np.reshape(trial, (trial.shape[0] * trial.shape[1], 
                                trial.shape[2]))
trial_id   = np.reshape(trial_id, (trial_id.shape[0] * trial_id.shape[1], 
                                   trial_id.shape[2]))
trial_resp = np.reshape(trial_resp, (trial_resp.shape[0] * trial_resp.shape[1], 
                                     trial_resp.shape[2]))

logging.info('INFO:\tData reshaped in long format')

# split data in blocks of 32 trials
tr_amount  = len(trial)
trial      = np.split(trial, tr_amount // 32)
trial_id   = np.split(trial_id, tr_amount // 32)
trial_resp = np.split(trial_resp, tr_amount // 32)

logging.info('INFO:\tData split in appropriate block length')

#%%

# -------------------- #
# PSYCHOPY DEFINITIONS #
# -------------------- #

warning_font = 'firebrick'
main_font    = 'black'    
relax_font   = rel_font
focus_font   = foc_font

# initialize the window
win = visual.Window(fullscr = full_screen, 
                    size    = (1000, 800), 
                    units   = "norm", 
                    color   = 'white')

# initialize text elements
welcome      = visual.TextStim(win, 
                               text   = "Welkom!\n\nDruk op de spatie om verder te gaan", 
                               color  = main_font)
instructions = visual.TextStim(win, 
                               text   = "OK", 
                               height = 0.075, 
                               color  = main_font)
exercise     = visual.TextStim(win, 
                               text   = "OK", 
                               height = 0.075, 
                               color  = main_font)
seeing_stim = visual.TextStim(win, 
                               text   = "OK", 
                               height = 0.075, 
                               color  = main_font)
stim_short  = visual.TextStim(win, 
                               text   = "OK", 
                               height = 0.075, 
                               color  = main_font)
count_down   = visual.TextStim(win, 
                               text   = "OK", 
                               height = 0.075, 
                               color  = main_font)
fixation     = visual.TextStim(win, 
                               text   = "OK", 
                               height = 0.1)
not_answered = visual.ImageStim(win, 
                               image = stim_dir + '\\not_answered.jpg')
answered     = visual.ImageStim(win, 
                               image = stim_dir + '\\answered.jpg')
anticipation = visual.TextStim(win, 
                               text   = "OK", 
                               color  = main_font)
feedback     = visual.TextStim(win, 
                               text   = "OK", 
                               color  = main_font)
inter_block  = visual.TextStim(win, 
                               text   = "OK", 
                               color  = main_font)
pressed_quit = visual.TextStim(win, 
                               text   = "OK", 
                               color  = main_font)
goodbye      = visual.TextStim(win, 
                               text   = "Bedankt voor jouw deelname!\n\n\n" +
                                        "Geef een seintje aan de experimentleider", 
                               height = 0.1, 
                               color  = main_font)
speedy_mode  = visual.TextStim(win, 
                               text   = "WAARSCHUWING: Speedy mode is actief\n\n" + 
                                        "Druk op een toets om verder te gaan", 
                               height = 0.1, 
                               color  = warning_font)
block_end    = visual.TextStim(win, 
                               text   = "Einde van dit blok\n\nDruk op de spatie om verder te gaan", 
                               height = 0.1, 
                               color  = main_font)

# ********************* #
# defining instructions #
# ********************* #

seeing_stim.text   = ("Om er voor te zorgen dat je vertrouwd bent met de figuren " +
                      "die je zal zien tijdens dit blok zullen we deze eerst even tonen aan jou.\n\n"
                      "Neem jouw tijd om te figuren goed te bekijken.\n\n"
                      "Wanneer je de figuren goed hebt gezien mag je op de spatie drukken om verder te gaan.")

stim_short.text    = ("Kijk zolang naar de figuren als je zelf wil.\n\n"
                      "Druk op de spatie om verder te gaan.")

pressed_quit.text  = ("LOG: [escape] toets werd ingedrukt.\n" +
                     "\nDruk op [ q ] om te stoppen met dit experiment.\n" +
                     "\nDruk op [ c ] om verder te gaan met het volgende blok.")

#%%

# ---------------- #
# START EXPERIMENT #
# ---------------- #

win.mouseVisible = False

# display a warning when speedy mode is activated
if speedy:
     speedy_mode.draw()
     win.flip()
     answer  = event.waitKeys(keyList = ["c","q"])
     if answer[0] == 'q':
         speedy = False

# display the welcome message
welcome.draw()
win.flip()
event.waitKeys(keyList = "space")

colors     = ['zwart', 'blauw']
if int(random_param[subject_number - 1][1]) == 1:
    colors = colors[::-1]
    
instr_parts = ["In dit experiment zal je verschillende figuren na elkaar zien.\n\n" +
               "Jouw doel is om te bepalen op welke toets je moet drukken bij iedere figuur.",
         
               "Je kan de volgende twee toetsen gebruiken:\n[ f ] en [ j ].\n\n" +
               "Gebruik je linker wijsvinger om de [ f ] toets in te drukken.\n\n" +
               "Gebruik je rechter wijsvinger om de [ j ] toets in te drukken.",
         
               "Om te bepalen welke toets bij iedere figuur past zal je moeten proberen.\n\n" +
               "Probeer te onthouden op welke toets je moet drukken bij iedere figuur, dit omdat " +
               "de proefpersoon die het beste presteert nog een extra beloning krijgt.", 
               
               "Een voorlaatste puntje voor we starten met het experiment is het kruisje waar je naar kijkt tijdens het experiment.\n\n" +
               "Afhankelijk van de situatie zal het kruisje een andere kleur hebben.\n\n" + 
               "Zo zal het kruisje {0} zijn als je even moet opletten, en {1} als er even niks zal gebeuren.".format(colors[0], colors[1]),
               
               "Een {0} kruisje duidt aan dat er zeer binnenkort een figuur zal verschijnen waar je op kan reageren.\n\n".format(colors[0]) +
               "Een {0} kruisje duidt dan weer aan dat je tussen twee trials in zit.\n\n".format(colors[1]) + 
               "Probeer gedurende het experiment te focussen op het kruisje, ongeacht de kleur.\nMooie data =  ik blij :)",
    
               "Tot slot: rond de figuur zal telkens een zwart kader te zien zijn.\n\n" + 
               "Wanneer je op een toets drukt zal het kader grijs worden.\n\n" +
               "Door hier op te letten ben je zeker dat je effectief gedrukt hebt!"]
    
exer_parts  = ["Om je vertrouwd te maken met het verloop van dit experiment laten we jou eerst een oefenronde doorlopen.\n\n" +
               "De bedoeling hiervan is om ervoor te zorgen dat je het experiment goed begrijpt.\n\n"
               "De oefenronde duurt twee blokken, daarna starten we met het eigenlijke experiment.",
               
               "Wat je vooral moet onthouden is dat er maar twee antwoordknoppen zijn:\n[ f ] en [ j ].\n\n" +
               "Gebruik je linker wijsvinger om de [ f ] toets in te drukken.\n\n" +
               "Gebruik je rechter wijsvinger om de [ j ] toets in te drukken.", 
               
               "Let verder ook op de kleur van het kruisje:\n\n" +
               "- {0} als je even moet opletten.\n- {1} als er even niks zal gebeuren.".format(colors[0].capitalize(), colors[1].capitalize()), 
               
               "Probeer te bepalen welke toets je moet indrukken bij iedere figuur door te proberen.\n"]

inter_parts = ["De oefenronde zit er op.\n\n"
               "Je kan een korte pauze inlassen als je dit wenst.\n\n", 
               
               "Als je nog vragen hebt over dit experiment kan je deze nu stellen aan de proefleider.",
               
               "Druk op de spatie toets om te starten met het eerste blok van het experiment."]

# display the instructions
for instr in instr_parts:
    instructions.text = instr
    instructions.draw()
    win.flip()
    event.waitKeys(keyList = "space")

#%%

# array where the experiment data is stored
saved_data = np.zeros((tr_amount, 7))

# store some data already
saved_data[:,:3]  = np.reshape(trial_id, (tr_amount, 3))
saved_data[:,3:4] = np.reshape(trial_resp, (tr_amount, 1))

# split saved_data in appropriate format
saved_data = np.split(saved_data, tr_amount // 32)

#%%

# ---------------------- #
# START EXPERIMENT BLOCK #
# ---------------------- #

exercising      = True
exercise_saved  = False
exercise_blocks = np.arange(0, no_exercise_blocks * 2)

# initial definition of too_slow
too_slow        = False

# triggers: initialize
if EEG_measure:
    send_triggers(init    = True, 
                  sending = False)

for block in range(len(trial)):
    
    # install security measure
    break_by_esc = False
    
    # draw specific message depending on progress
    if block == 0:
        for instr in exer_parts:
            exercise.text = instr
            exercise.draw()
            win.flip()
            event.waitKeys(keyList = "space")
        # start of the experiment
        if EEG_measure:
            send_triggers(init    = False, 
                          sending = True, 
                          value   = 1)
    elif block == (exercise_blocks[-1] + 1):
        for instr in inter_parts:
            inter_block.text = instr
            inter_block.draw()
            win.flip()
            event.waitKeys(keyList = "space")
    else:
        pass
    
    # different screens (not) shown depending on speedy mode
    if speedy:
        pass
    else:
        # some explanation about seeing the stimuli
        if block in exercise_blocks:
            seeing_stim.draw()
        else:
            stim_short.draw()
        win.flip()
        event.waitKeys(keyList = "space")
    
    # first show the stimuli that they will see
    used_stimuli = np.unique(trial_id[block][:,0] + np.array(1))
    locs         = np.array([-.4, -.1, .2, .5])
    np.random.shuffle(locs)
    os.chdir(stim_dir)
    
    for i in range(len(used_stimuli)):
        stim_name = 'Figuur' + str(int(used_stimuli[i])) + '.jpg'
        shown = visual.ImageStim(win, 
                                 image = stim_name)
        shown.pos = (0, locs[i])
        shown.draw()

    win.flip()
    
    # triggers: seeing stimuli
    if EEG_measure:
        send_triggers(init    = False, 
                      sending = True, 
                      value   = 40) 
    
    if speedy:
        core.wait(speedy_wait)
    else:
        logging.info('INFO:\tSeeing stimuli')
        event.waitKeys(keyList = "space")
        
        # triggers: seeing stimuli
        if EEG_measure:
            send_triggers(init    = False, 
                          sending = True, 
                          value   = 41) 
        
    # counting down until the start of the block
    if speedy:
        pass
    else:
        for seconds in range(5, 0, -1):
            count_down.text = "{0:d} seconden tot de start van het blok".format(seconds)
            count_down.draw()
            win.flip()
            core.wait(1)
    
    from timeit import default_timer as timer
    
    # loop over all trials within an experiment block
    for tr in range(len(trial_id[block])):
        
        # start of this trial
        if EEG_measure:
            send_triggers(init    = False, 
                          sending = True, 
                          value   = 50) 
            
        start = timer()
        
        logging.critical("CRITICAL:\tSTART: Trial {0:02d} - block {1:02d}".format(tr+1, block+1))
        
        # jittering: fixation cross
        if speedy:
            fixation_wait = speedy_wait
        else:
            fixation_wait = jitter[block][tr,0] / 1000
        
        # indicate that the participants should focus now
        fixation.text  = "+"
        fixation.color = foc_font
        fixation.draw()
        win.flip()        
        core.wait(fixation_wait)
        
        # determine which stimulus is shown in this trial
        row_indx    = int(trial_id[block][tr,0]) + 1
        stim_name   = 'Figuur' + str(row_indx) + '.jpg'
        
        # cd to stimulus directory
        os.chdir(stim_dir)
        
        # define the stimuli used
        stimulus   = visual.ImageStim(win, 
                                      image = stim_name)
        
        # actually draw the stimuli
        not_answered.draw()
        stimulus.draw()
        win.flip()
        
        # triggers: stimulus locked
        if EEG_measure:
            if int(np.unique(trial_id[block][:,2])) == 1:
                send_triggers(init    = False, 
                              sending = True, 
                              value   = 60) 
            else:
                send_triggers(init    = False, 
                              sending = True, 
                              value   = 61) 
               
        # clear the keyboard input
        event.clearEvents(eventType = "keyboard")
        
        # Wait for the response
        if speedy:
            start_time = 0
            stop_time  = np.random.randint(250, 1000) / np.random.randint(901, 1000)
            keys       = np.random.choice(["f","j", None])
        else:
            too_slow   = True
            start_time = clock.getTime()
            keys       = event.waitKeys(keyList = ["f","j", "escape"],
                                        maxWait = max_resp_time)
            stop_time = clock.getTime()
        
        if keys is None:
            
            # triggers: response locked - no answer
            if EEG_measure:
                send_triggers(init = False, 
                              sending = True,
                              value   = 72) 
                    
            reaction_time = max_resp_time * 1000
            resp          = np.array([-1.])           # no response
            accurate      = False
        else:
                   
            # update the rectange color
            answered.draw()
            stimulus.draw()
            win.flip()
            
            # change too_slow:
            too_slow = False
            
            # reaction times
            reaction_time = (stop_time - start_time) * 1000
           
            # determine key press
            if keys[0] == 'f':
                resp = np.array([0.])
            elif keys[0] == 'j':
                resp = np.array([1.])
            else:
                logging.critical('CRITICAL:\t[escape] key used')
                break_by_esc = True
                break
            
            # compute accuracy
            accurate = np.array_equal(trial_resp[block][tr,:], resp)
            
            # triggers: response locked - in time
            if EEG_measure:
                if accurate:
                    send_triggers(init    = False, 
                                  sending = True, 
                                  value   = 70)      # correct response
                else:
                    send_triggers(init    = False, 
                                  sending = True, 
                                  value   = 71)      # incorrect response
            
            # show the stimulus for the rest of the period
            if speedy:
                core.wait(speedy_wait)
            else:
                core.wait(max_resp_time - (stop_time - start_time))
            win.flip(clearBuffer=True)
            
        # anticipation for the reward
        if speedy:
            anticipation_wait = speedy_wait
        else:
            anticipation_wait = jitter[block][tr,1] / 1000
        
        anticipation.text = '...'
        anticipation.draw()
        win.flip()
        logging.info('INFO:\t\tAnticipation phase')
        
        core.wait(anticipation_wait)
               
        if too_slow:
            feedback.text = "Te traag"
        else:
            if accurate:
                feedback.text = "Correct antwoord"
            else:
                feedback.text = "Fout antwoord"
                
        # write data to array
        saved_data[block][tr,4:] = np.array([resp, accurate, reaction_time])
            
        # jittering: feedback
        if speedy:
            feedback_wait = speedy_wait
        else:
            feedback_wait = jitter[block][tr,2] / 1000
        
        feedback.draw()
        win.flip()
        
        # triggers: feedback locked
        if EEG_measure:
            if too_slow:
                send_triggers(init     = False, 
                              sending  = True,
                              value    = 82) 
            else:
                if accurate:
                    send_triggers(init = False, 
                              sending  = True,
                              value    = 80) 
                else:
                    send_triggers(init = False, 
                              sending  = True,
                              value    = 81) 
        
        # let feedback stay on screen
        core.wait(feedback_wait)
        
        # end of the feedback
        if EEG_measure:
            send_triggers(init = False, 
                                  sending  = True,
                                  value    = 85) 
        
        # give the participants a break
        fixation.text  = "+"
        fixation.color = rel_font
        fixation.draw()
        win.flip()      
        
        if speedy:
            relax_wait = speedy_wait
        else:
            relax_wait = jitter[block][tr,3] / 1000
        core.wait(relax_wait)
        
        # write data to temporary file
        writing_to = r'experiment\temp'
        prefix     = 'exp'
        if exercising:
            writing_to = r'exercise\temp'
            prefix     = 'exc'

        folder_search(path = subject_dir, make_dirs = False, cd_to = writing_to)
        temp_data = np.reshape(saved_data, (tr_amount, 7))
        np.save('{0:s} - sub-{1:02d}_ses-01_task-overtraining - temp.npy'.format(prefix, subject_number), temp_data)
      
        logging.critical("CRITICAL:\tEND: Stim {0:03d} - rep {1:02d} - trial {2:02d} of block {3:02d}".format(row_indx,
                                                                                                              int(trial_id[block][tr,1]),
                                                                                                              tr+1, 
                                                                                                              block+1, 
                                                                                                              ))

        end = timer()
        print('Elapsed time: ', end - start)

    if break_by_esc:
        if exercising:
            folder_search(path = subject_dir, make_dirs = False, cd_to = r'exercise')
            exc_data = np.reshape(saved_data, (tr_amount, 7))[:no_exercise_blocks*64,:]
            np.save('exc - sub-{0:02d}_ses-01_task-overtraining.npy'.format(subject_number), exc_data)
            logging.critical('CRITICAL:\t[BREAK]: Data exercise phase safely written away')
        else:
            folder_search(path = subject_dir, make_dirs = False, cd_to = r'experiment')
            exp_data = np.reshape(saved_data, (tr_amount, 7))[no_exercise_blocks*64:,:]
            np.save('exp - sub-{0:02d}_ses-01_task-overtraining.npy'.format(subject_number), exp_data)
            logging.critical('CRITICAL:\t[BREAK]: Data experiment phase safely written away')   
            pressed_quit.draw()
        
        pressed_quit.draw()
        win.flip()
        keys  = event.waitKeys(keyList = ["q","c"])
        if keys[0] == "c":
            continue
        else:
            break       
    
    if (block + 1 == 2) and not exercise_saved:
        folder_search(path = subject_dir, make_dirs = False, cd_to = r'exercise')
        exc_data = np.reshape(saved_data, (tr_amount, 7))[:no_exercise_blocks*64,:]
        np.save('exc - sub-{0:02d}_ses-01_task-overtraining.npy'.format(subject_number), exc_data)
        logging.critical('CRITICAL:\tData exercise phase safely written away')
        exercise_saved = True

    if speedy:
        # mark the end of a block
        block_end.draw()
        win.flip()
        core.wait(speedy_wait)
    else:
        # mark the end of a block
        if EEG_measure:
            send_triggers(init    = False, 
                          sending = True, 
                          value   = 100) 
        block_end.draw()
        win.flip()
       
        # wait for a response by the participant
        event.waitKeys(keyList = "space")
        send_triggers(init    = False, 
                      sending = True, 
                      value   = 101)
    
    # clear the keyboard input
    event.clearEvents(eventType = "keyboard")
    

#%%
            
# --------- #
# SAVE DATA #
# --------- #

# experiment data  
folder_search(path = subject_dir, make_dirs = False, cd_to = r'experiment')
exp_data = np.reshape(saved_data, (tr_amount, 7))[no_exercise_blocks*64:,:]
np.save('exp - sub-{0:02d}_ses-01_task-overtraining.npy'.format(subject_number), exp_data)
logging.critical('CRITICAL:\tAll data safely written away')

#%%
        
# ----------------- #
# END OF EXPERIMENT #
# ----------------- #

# display the goodbye message
goodbye.draw()
win.flip()

# end experiment
if EEG_measure:
    send_triggers(init    = False, 
                  sending = True, 
                  value   = 2)

quit_exp = event.waitKeys(keyList = ["q"])

# close the experiment window
win.close()

# close logging file
logging.critical('CRITICAL:\tExperiment succeeded')
logging.shutdown()
