##################################################
## Project: Theta and alpha power across fast and slow timescales in cognitive control
## Script purpose: follow-up neural analyses (see 'Correlation between neural and behavioral metrics')
## Date: 2020
## Author: Pieter Huycke
##################################################

# SETUP + PREPARING DATA ---------
setwd("C:/Users/pieter/OneDrive - UGent/Projects/2019/overtraining - PILOT 3/figures/Publish/Data/Stimulus-locked/Theta, alpha, beta + behavioral data")
df = read.csv("theta_alpha_beta_behavioural.csv", 
              header = TRUE, sep = ",")

# check if data load worked out
head(df)

# load needed libraries
library(car)
library(lme4)
library(lmerTest)

# VARIABLE MANIPULATION ---------

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

# Are alpha and theta predictors of RT? ---------

# caveat: this is the full data
# -- > correlation exists between repetitions_block and block
rt.all = lmer(RT_log  ~ (1 + Theta + Alpha|Subject_nr) + Theta + Alpha +
                         Repetitions_block +  Repetitions_block:Condition +
                         Block_specific * Condition, data = df)

# select the best subset of variables that also leads to decent performance
rt_all.selected = lmerTest::step(rt.all)
rt_all.selected
rt_all.final    = lmerTest::get_model(rt_all.selected)
anova(rt_all.final)
summary(rt_all.final)

# show whether correlations exist between the variables using Variance Inflation Factor
# all lower than 2, so not too much correlation
all_vifs = car::vif(rt_all.final)
print(all_vifs)
