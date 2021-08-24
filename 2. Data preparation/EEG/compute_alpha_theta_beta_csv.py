
import numpy as np
import os
import pandas as pd
import mne

#%%

def get_power_data(epoch       : mne.epochs.EpochsFIF,
                   freq_arr    : np.ndarray,
                   condition   : str,
                   electrodes  : str = "eeg",
                   stim_locked : bool = True) -> np.ndarray:
    """
    Returns the baselined power data for a certain condition. Can be run on 
    separate participants to allow permutation testing.

    Parameters
    ----------
    epoch : mne.epochs.EpochsFIF
        An Epochs object to run the TF on.
    freq_arr : np.ndarray
        An array containing the frequencies to consider in the TF. A list is
        acceptable too.
    condition : str
        A string indicating a condition based on which the Epoch object is 
        subsetted.
    electrodes : str, optional
        The electrode selection on which to run TF. The default is "eeg".
    stim_locked : bool, optional
        Will return a different set of the power values. The default is True.
        
    Returns
    -------
    np.ndarray
        An array of the form electrodes times frequencies times timepoints
        containing the power values.
    """    
    
    # get the TFR
    passed = epoch
    if condition is not None:
        passed = epoch[condition]
        
    power = mne.time_frequency.tfr_morlet(passed, 
                                          freqs      = freq_arr,
                                          n_cycles   = freq_arr / 2,   # or 6
                                          return_itc = False, 
                                          verbose    = False,
                                          picks      = electrodes,
                                          n_jobs     = 4)
    
    if stim_locked:
        # is used when the data is stimulus-locked
        power.apply_baseline((BASE_MIN, BASE_MAX), 
                             mode    = "logratio", 
                             verbose = False)
        after_baseline = np.where((power.times > 0) & (power.times <= 1))[0]
        return power.data[:,:,after_baseline]
    else:
        unbiased = np.where((power.times > -.235) & (power.times <= .305))[0]
        return power.data[:,:,unbiased]


def external_baseline(arr_as_base : np.ndarray,
                      arr_to_base : np.ndarray,
                      arr_time    : np.ndarray,
                      baseline    : tuple = (-1, 0), 
                      keep_orig   : bool  = False) -> np.ndarray:
    """
    Takes an array with power values and a TFR, and baselines the inputted TFR
    with the provided array. Currently on "logratio" supported.

    Parameters
    ----------
    arr_as_base : np.ndarray
        The data adapted from the TFR of the Epoch that will be used as a 
        baseline.
    arr_to_base : np.ndarray
        The data adapted from the TFR of the Epoch we wish to baseline.
    arr_time : np.ndarray
        The time array, indicating the time in seconds for the Epoch. A list 
        will probably also work. Type stability is not enforced.
    baseline : tuple, optional
        The period in arr_as_base that will be used to baseline arr_to_base. 
        Time is expressed in seconds. The default is (-1, 0).
    keep_orig : bool, optional
        Will return the original data (i.e. no baseline applied), along with
        the baselined data. The default is False.
    
    Returns
    -------
    np.ndarray
        Returns arr_to_base that was baselined using the data between baseline 
        in arr_as_base. This array can then be used to replace TFR.data.
    """   
    
    # check whether the types match
    assert type(arr_as_base) == type(arr_to_base) == np.ndarray
    
    # make a copy of the original
    orig = arr_to_base.copy()
    
    # get the baseline values and the points in the array
    bmin, bmax = baseline
    i_min, i_max = int(np.where(arr_time == bmin)[0]), \
                   int(np.where(arr_time == bmax)[0]) + 1

    # compute the baselined power values
    average = np.mean(arr_as_base[..., i_min:i_max], 
                      axis     = -1,
                      keepdims = True)
    
    arr_to_base  /= average
    arr_baselined = np.log10(arr_to_base)

    if keep_orig:
        return orig, arr_baselined
    else:
        return arr_baselined
    
#%%

# Determines which part of the analysis to run
STIM_LOCKED = True

# these are the subjects that had all 512 epochs recorded and stored safely
full_epochs = ["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-08",
               "sub-10", "sub-12", "sub-13", "sub-15", "sub-16", "sub-17",
               "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
               "sub-25", "sub-26", "sub-27", "sub-28", "sub-29", "sub-30"]

# the lower- and upperbound for the baseline procedure
BASE_MIN = -1.00
BASE_MAX = -0.25

# get the alpha/theta power data
arrs = np.array([])

# Condition and frequency band constants, influence analysis done
ALPHA_BAND = True
CONDITION  = "All"            # Novel, Recurring or All (no subset)

#%%

"""
Subject specific data
Here we store all the epoch data for each subject in a list. This was done so
that the data for each subject can be accessed to do e.g. TF on this data. The
data is also merged into one single Epoch file, which can be used to do an 
overall analysis (e.g. the TF representation of block 1 over subjects).
"""

# the root folder, but split up for readability
general  = r"C:\Users\pieter\OneDrive - UGent\Projects\2019"
specific = r"\overtraining - PILOT 3"
path     = general + specific

DATA     = os.path.join(path, "figures", "TF", "Group level", "data")
FIG      = os.path.join(path, "figures", "TF", "Group level", "plots")

del general, specific

#%%

"""
Subject specific data
Here we store all the epoch data for each subject in a list. This was done so
that the data for each subject can be accessed to do e.g. TF on this data. The
data is also merged into one single Epoch file, which can be used to do an 
overall analysis (e.g. the TF representation of block 1 over subjects).
"""

epochs_list = []
ROOT = r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish"

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

#%%

# replace the spaces in the metadata titles with _
metadata = epoch_all.metadata
for name in metadata.columns:
    metadata = metadata.rename(columns = {name: name.replace(" ","_")})
epoch_all.metadata = metadata

#%%

# load the correct data
if STIM_LOCKED:
    DATA = os.path.join(ROOT, "Data", "Stimulus-locked")
    if ALPHA_BAND:
        arrs  = np.load(os.path.join(DATA,
                                     "Repetition 1, 2, ... 8",
                                     "stim_alpha_repetitions_rs.npy"))
    else:
        arrs  = np.load(os.path.join(DATA, 
                                     "Block 1, 2, ... 8",
                                     "stim_theta_blocks_rs.npy"))

# transform the data into a readable pandas dataframe
df    = pd.DataFrame(arrs)
df[1] = full_epochs * (len(df) // len(full_epochs))
df[2] = list(np.repeat(np.arange(1, 9), 24)) * 2
df[3] = np.repeat(["Novel", "Recurring"], len(arrs) // 2)

# set the column names
if STIM_LOCKED:
    if ALPHA_BAND:
        df.columns = ["Alpha power", "Subject number", "Repetition count", "Condition"]
    else:
        df.columns = ["Theta power", "Subject number", "Block number", "Condition"]

# --------------------------------- #
# Load and process behavioral  data #
# --------------------------------- #

# read the behavioral data
os.chdir(r"C:\Users\pieter\Downloads\GitHub\phuycke\PhD-code\EEG\Extensive training project\Analysis\Behavioral\General\Data")
b = pd.read_pickle("behavioral_clean.pkl")

# change the behavioral data for easier processing
b = b.drop(['Trial number overall', 'Trial number block', 'Block number general', 
            'Overall repetitions', 'Stimulus ID', 'Response', 'Correct'], 
           axis = 1)
b.columns      = ["Condition", "Block number", "Repetition count", 
                  "Reaction times", "Subject number"]
b["Condition"] = b["Condition"].replace({0: "Novel", 1: "Recurring"})
b              = b[['Reaction times', 'Subject number', 'Repetition count', 
                    'Condition', 'Block number']]

# delete the subjects not in 'full_epochs'
nums = [(int(name[-2:])) for name in full_epochs]
b    = b[b['Subject number'].map(lambda x: x in nums)]

# finally get the needed pickle
var_string = "Repetition count"
if not ALPHA_BAND:
    var_string = "Block number"
df_beh = b.groupby([var_string, 'Subject number',"Condition"], 
                   as_index=False).agg({'Reaction times':np.mean})
df_beh["Subject number"] = ["sub-{0:02d}".format(s) for s in df_beh["Subject number"]]
df_beh = df_beh[["Reaction times", "Subject number", 
                 var_string, "Condition"]]

# ------------------------------------- #
# Prepare data for correlation analysis #
# ------------------------------------- #

# remove unneeded variables to avoid clutter
del b

# create a temporary array
temp = df
temp["Reaction times"] = np.nan

# match each row of the behavioral data with the corresponding row in the neural data
# get the RT from the behavioral array, and past it in the temp dataframe
# TODO: optimize this, because this function is slow due to lookup
for index, row in temp.iterrows():
    # get the specifics from each row
    t1, t2, t3 = row["Subject number"], row[var_string], row["Condition"]
    # get the index to refer to later
    i1 = temp.loc[(temp['Subject number'] == t1) & \
                  (temp[var_string] == t2) & \
                  (temp['Condition'] == t3)].index
    # get the index for the behavioral data, given the cols in neural
    i2 = df_beh.loc[(df_beh['Subject number'] == t1) & \
                    (df_beh[var_string] == t2) & \
                    (df_beh['Condition'] == t3)].index
    # get the correct RT, and write it to the temp DF
    temp.loc[i1, "Reaction times"] = float(df_beh.loc[i2,'Reaction times'])

# vary which type of power is checked
var_pow = "Alpha power"
if not ALPHA_BAND:
    var_pow = "Theta power"

# rearrange the columns for convenience
temp = temp[[var_pow,'Reaction times','Subject number',var_string,'Condition']]

# remove rows containing at least one NaN, standarize power and save in new variable
temp = temp.dropna()
temp[var_pow] = (temp[var_pow] - temp[var_pow].mean()) / temp[var_pow].std()
temp["Reaction times"] = (temp["Reaction times"] - temp["Reaction times"].mean()) / temp["Reaction times"].std()

# subset the data based on the condition we are interested in
if CONDITION != "ALL":
    temp = temp[temp["Condition"] == CONDITION].copy(deep = True)

#%%

"""
In this block, we attempt to compute the theta, alpha and beta power for
each trial separately. The time window used to compute the theta power is
deduced from the permutation test results (200 - 350 ms after stimulus onset),
the same applies for the time window of interest of the alpha power (700 -
850 ms after stimulus onset)
The beta power was never researched, but we assume that it is situated in the 
same time windows as the alpha activation based on the permutation test results
Therefore, we will look beta activation in this same time window
"""

# define frequencies of interest
frequencies = np.logspace(np.log10(4), 
                          np.log10(30), 
                          15)

# store the needed power arrays for each subject separately
all_dat    = np.zeros((metadata.shape[0], metadata.shape[1]+3))
freq_range = [frequencies[:5], frequencies[5:9], frequencies[9:]]
time_range = [(.2, .35), (.7, .85), (.7, .85)]
electrodes = ["Fz", "Pz", "Pz"]

# select the data we are interested in
times      = epochs_list[0].times
times      = times[np.where((times > 0) & (times <= 1))[0]]

# load the data to apply the external baseline if needed
if not STIM_LOCKED:
    stimulus_data = np.load(r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data\stimulus_tfr.npy")
    stimulus_time = np.load(r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\TF\Group level\data\stimulus_times.npy")

print("\n****")
print("Saving theta, alpha and beta power for each trial")
print("Saving all available behavioral data for each trial")
print("***\n")

print("Progress:")
for i in range(len(epoch_all)):
    for j in range(len(freq_range)):
        f        = freq_range[j]
        t        = time_range[j]
        indices  = np.where((times > t[0]) & (times <= t[-1]))[0]
        pow_vals = []
        if STIM_LOCKED:
            power = get_power_data(epoch       = epoch_all[i],
                                   freq_arr    = freq_range[j],
                                   condition   = None,
                                   electrodes  = electrodes[j],  # Note: 1 electrode
                                   stim_locked = STIM_LOCKED)
            all_dat[i,j] = np.mean(power[:,:,indices])
        else:
            power = get_power_data(epoch       = epoch_all[i],
                                   freq_arr    = frequencies,
                                   condition   = "Block_ID == {} and Block_number_specific == {}".format(i, k),
                                   stim_locked = STIM_LOCKED)
            bslnd = external_baseline(arr_as_base = stimulus_data,
                                      arr_to_base = power,
                                      arr_time    = stimulus_time,
                                      baseline    = (-1, -.5), 
                                      keep_orig   = False)
            bslnd = bslnd[:,np.where((frequencies > 8) & (frequencies <= 13))[0],:]
            all_dat[i,j] = np.mean(bslnd[:,:,indices])
    all_dat[i,j+1:] = epoch_all[i].metadata
    if i in np.round(np.linspace(0, len(epoch_all)-1, 100)):
        percent = np.where(i == np.round(np.linspace(0, len(epoch_all)-1, 100)))
        print("--> {0:3.0f}% completed".format(percent[0][0]))
        
# save the data 
os.chdir(r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data")
np.save("theta_alpha_beta_behavioural.npy", all_dat)
 
#%%

os.chdir(r"C:\Users\pieter\OneDrive - UGent\Projects\2019\overtraining - PILOT 3\figures\Publish\Data\Stimulus-locked\Theta, alpha, beta + behavioral data")
all_dat = np.load("theta_alpha_beta_behavioural.npy")

# convert to pandas dataframe to use in R
all_dat_pd = pd.DataFrame(all_dat)
all_dat_pd.columns = ['Theta', 'Alpha', 'Beta', 'Trial_overall', 'Trial_block',
                      'Block_overall', 'Condition', 'Block_specific',
                      'Repetitions_overall', 'Repetitions_block', 
                      'Stimulus_ID', 'Response', 'Accuracy_int', 'RT', 
                      'Subject_nr']

# replace binary values with interpretable labels
# add extra variables based on transformations of existing data
all_dat_pd["Condition"]      = all_dat_pd["Condition"].replace({0: "Novel", 
                                                                1: "Recurring"})
all_dat_pd["Error_int"]      = np.abs(all_dat_pd["Accuracy_int"] - 1)
all_dat_pd["Accuracy_label"] = all_dat_pd["Accuracy_int"].replace({0: "Wrong",
                                                                   1: "Correct"})
all_dat_pd["Response" ]      = all_dat_pd["Response"].replace({0: "Left", 
                                                               1: "Right"})
all_dat_pd["RT_log"]         = np.log10(all_dat_pd["RT"])

# change the order of the columns for convenience
all_dat_pd = all_dat_pd[['RT', 'RT_log', 'Accuracy_label', 'Accuracy_int',
                         'Error_int', 'Theta', 'Alpha', 'Beta', 'Subject_nr', 
                         'Repetitions_overall', 'Repetitions_block', 
                         'Block_overall', 'Block_specific', 'Condition', 
                         'Trial_overall', 'Trial_block', 'Response', 
                         'Stimulus_ID']]

# save as a csv to load in R
all_dat_pd.to_csv("theta_alpha_beta_behavioural.csv", 
                  sep    = ",", 
                  header = True, 
                  index  = False)
