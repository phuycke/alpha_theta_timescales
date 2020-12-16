##################################################
## Project: Theta and alpha power across fast and slow timescales in cognitive control
## Script purpose: follow-up neural analyses (see 'Alpha-theta Dissociation' in EEG Results)
## Date: 2020
## Author: Pieter Huycke
##################################################

# SETUP + PREPARING DATA ---------
setwd("C:/Users/pieter/OneDrive - UGent/Projects/2019/overtraining - PILOT 3/figures/Publish/Data/Stimulus-locked/Theta, alpha, beta + behavioral data")
df = read.csv("theta_alpha_beta_behavioural.csv", 
              header = TRUE, sep = ",")

# load needed libraries
library(car)
library(lme4)

# VARIABLE MANIPULATION  ---------

# see as certain predictors as factor
df$Block_overall       = factor(df$Block_overall)
df$Block_specific      = factor(df$Block_specific)
df$Repetitions_overall = factor(df$Repetitions_overall)
df$Repetitions_block   = factor(df$Repetitions_block)
df$Subject_nr          = factor(df$Subject_nr)

# check the changes
class(df$Block_overall)
class(df$Block_specific)
class(df$Repetitions_overall)
class(df$Repetitions_block)
class(df$Subject_nr)

# set the Anova parameters
options(contrasts = c("contr.sum", "contr.poly"))

# Alpha-theta dissociation ---------

library(dplyr)

# keep only the theta power metric, rename Theta to Power, and add dummy that codes for power
theta = select(df, Subject_nr, Theta, Condition, Block_specific, Repetitions_block)
names(theta)[names(theta)=="Theta"] = "Power"
theta$Power_dummy = 0

# keep only the alpha power metric, rename Alpha to Power, and add dummy that codes for power
alpha = select(df, Subject_nr, Alpha, Condition, Block_specific, Repetitions_block)
alpha$Power_dummy = 1
names(alpha)[names(alpha)=="Alpha"] = "Power"

# bind the two dataframes together
df_powersplit = rbind(theta, alpha)
head(df_powersplit)

# -------------- #
# Fast timescale #
# -------------- #
analysis.1 = lmer(Power  ~ (1|Subject_nr) + Repetitions_block * Condition * Power_dummy, 
                  data = df_powersplit)
summary(analysis.1)
dissociation.fast = Anova(analysis.1,
                          type           = "III",
                          test.statistic = "F")
dissociation.fast

# -------------- #
# Slow timescale #
# -------------- #
analysis.2 = lmer(Power  ~ (1|Subject_nr) + Block_specific * Condition * Power_dummy, 
                  data = df_powersplit)
summary(analysis.2)
dissociation.slow = Anova(analysis.2,
                          type           = "III",
                          test.statistic = "F")
dissociation.slow
