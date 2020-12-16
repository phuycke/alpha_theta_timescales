# Theta and alpha power across fast and slow timescales in cognitive control

[![Up to date](https://img.shields.io/github/last-commit/phuycke/alpha_theta_timescales)](https://img.shields.io/github/last-commit/phuycke/alpha_theta_timescales)
[![Lines of code](https://img.shields.io/tokei/lines/github/phuycke/alpha_theta_timescales?color=informational)](https://img.shields.io/tokei/lines/github/phuycke/alpha_theta_timescales?color=informational)
[![Follow me](https://img.shields.io/twitter/follow/PieterHuycke?style=social)](https://img.shields.io/twitter/follow/PieterHuycke?style=social)

## Overview

This GitHub repository contains the code used for the analysis and plotting done in the paper titled "Theta and alpha power across fast and slow timescales in cognitive control" by Pieter Huycke, Pieter Verbeke, C. Nico Boehler and Tom Verguts. A preprint of this article is available on [BioRxiv.](https://doi.org/10.1101/2020.08.21.259341) The code available in this repository relies on the data collected for our study. This data is also open access, and can be found in this [Open Science Framework repository](https://osf.io/2q5eh/) when the article is published. 

## Organization

The scripts are organized following the order defined in [the paper](https://doi.org/10.1101/2020.08.21.259341).

- 0. Software environment
    * A list of all the softwares used in the Anaconda environment, and their version at the moment of submission
- 1. Experiment
    * Code
        - The experiment code itself. Note that ```main (only sub-01).py``` is a version that was used for the first subject. This version was too long, and hence we used a shorter version for the other 29 subjects ```main.py```.
    * Stimuli
        - The stimuli used in this experiment. Provided to us by [Hannes Ruge](https://doi.org/10.1093/cercor/bhp228). 
- 2. Data preparation
    * Behavioral
        - Behavioral data rejection (```clean.py```)
        - Definition of our descriptive statistics (```descriptives.py```)
    * EEG
        - Custom code written to preprocess our EEG data (```eeg_helper.py```)
        - The preprocessing script, described in 'Data Recording and Initial Processing' (```initial_processing.py```)
        - List of subject-specific rejected channels and rejected ICA components (```set_cleaning_parameters.py```)
- 3. Analysis
    * Ordered list of the analyses associated with the results described in the paper. The scripts are ordered so that their order corresponds with the order of the results described in the manuscript. Comments refer to the specific analysis done
- 4. Plots
    * Folders describe the specific figure, file names describe what specific part of the plot is created

## Programming environment   

Packages used, and their versions, are listed below.    
Note that we only listed the most important packages, a complete list of the Anaconda virtual environment can be found in [conda-env.txt.]()

- conda (v. 4.9.1)
- python (v. 3.8.5
    * matplotlib (v. 3.3.2)
    * mne-python (v. 0.21.0)
    * numpy (v. 1.19.2)
    * pandas (v. 1.1.3)
    * scipy (v. 1.5.0)
    * seaborn (v. 0.11.0)
    * spyder (v. 4.1.5)
- R (v. 3.6.3 - "Holding the Windsock")
- RStudio (v. 1.1.463)
    * lme4 (v. 1.1-25)
    * lmerTest (v. 3.1-3)

## Contact

- First author: Pieter Huycke  
    * [mail](mailto:Pieter.Huycke@UGent.be)
    * [web entry](https://www.cogcomneurosci.com/about/#pieter-huycke)
- Principle Investigator: prof. Tom Verguts
    * [mail](mailto:Tom.Verguts@UGent.be)
    * [web entry](https://www.cogcomneurosci.com/about/#principal-investigator)

[Lab website]: https://cogcomneurosci.com/

**Last edit: 16-12-2020**
