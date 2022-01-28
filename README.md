# Theta and alpha power across fast and slow timescales in cognitive control

[![Up to date](https://img.shields.io/github/last-commit/phuycke/alpha_theta_timescales)](https://img.shields.io/github/last-commit/phuycke/alpha_theta_timescales)
[![Follow me](https://img.shields.io/twitter/follow/PieterHuycke?style=social)](https://twitter.com/PieterHuycke)
[![DOI paper](https://img.shields.io/badge/Paper-https%3A%2F%2Fdoi.org%2F10.1111%2Fejn.15320-blue)](https://doi.org/10.1111/ejn.15320)
[![DOI data](https://img.shields.io/badge/Data-%20%20https%3A%2F%2Fdoi.org%2F10.5281%2Fzenodo.4659714-blue)](https://doi.org/10.5281/zenodo.4659714)

## Significance

This GitHub repository stores code and processed data files used for the paper published in [European Journal of Neuroscience](https://onlinelibrary.wiley.com/doi/epdf/10.1111/ejn.15320). The entire dataset (EEG + behavioral data) can be downloaded from [Zenodo](https://zenodo.org/record/4659714). 

## Folders

0. Software environment
    * Conda environment files (MNE, psychopy, analysis)
1. Experiment
    * Code: experiment code (PsychoPy3)
    * Stimuli: adopted with permission from [Ruge and Wolfensteller (2010)](https://doi.org/10.1093/cercor/bhp228). 
2. Data preparation
    * Behavioral: cleaning behavioral data + descriptive stats
    * EEG: custom EEG analysis pipeline + data cleaning
3. Analysis
    * Plotting (Python 3) + statistics (R) scripts. 
4. Plots
    * Scripts to recreate plots shown in the paper

## Programming environment   

**Python 3 Anaconda environment files**
- [PsychoPy3 (experiment run)](https://github.com/phuycke/alpha_theta_timescales/blob/main/0.%20Software%20environment/psychopy.yml)
- [EEG analysis (MNE)](https://github.com/phuycke/alpha_theta_timescales/blob/main/0.%20Software%20environment/mne.yml)
- [Data shaping and plotting](https://github.com/phuycke/alpha_theta_timescales/blob/main/0.%20Software%20environment/analysis.yml)

Unsure how to use these files? We refer to the [Anaconda3 documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

--- 

**R (v. 3.6.3 - "Holding the Windsock")**    
Extra packages needed:
- dplyr v. 1.02
- lme4 v. 1.1-26
- lmerTest v. 3.1-3

## Contact

Pieter Huycke (corresponding author)

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/ORCID_iD.svg/2048px-ORCID_iD.svg.png" alt="orcid logo" style="float:left;width:24px;height:24px;margin-right:5px"><a href="https://orcid.org/0000-0002-8156-4419">0000-0002-8156-4419</a>

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_cTJMGw0u-6pa22rOEAxiIXcaOaGp-_PXxw&usqp=CAU" alt="mail logo" style="float:left;width:24px;height:24px;margin-right:5px"><a href="mailto:pieter.huycke@hotmail.com?Subject=EJN%20Paper%202021">Mail me concerning this paper</a>

---

Tom Verguts (principle investigator)

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/ORCID_iD.svg/2048px-ORCID_iD.svg.png" alt="orcid logo" style="float:left;width:24px;height:24px;margin-right:5px"><a href="https://orcid.org/0000-0002-7783-4754">0000-0002-7783-4754</a>

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR_cTJMGw0u-6pa22rOEAxiIXcaOaGp-_PXxw&usqp=CAU" alt="mail logo" style="float:left;width:24px;height:24px;margin-right:5px"><a href="mailto:Tom.Verguts@UGent.be?Subject=EJN%20Paper%202021">Mail me concerning this paper</a>

---

<img src="https://cdn.pixabay.com/photo/2019/09/12/13/47/pictogram-4471660_1280.png" alt="website logo" style="float:left;width:24px;height:24px;margin-right:5px"><a href="https://www.cogcomneurosci.com/">Visit our lab website!</a>
