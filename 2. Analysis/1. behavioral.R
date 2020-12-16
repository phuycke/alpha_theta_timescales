##################################################
## Project: Theta and alpha power across fast and slow timescales in cognitive control
## Script purpose: behavioral analyses (see Behavioral Results)
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

# FIGURE 2A ---------

# make the subset so that total stimulus repetitions is smaller than 9
df.reduced = df[as.numeric(as.character(df$Repetitions_overall)) <= 8, ]

# ------------------------------ #
# log(RT) ~ stimulus repetitions #
# ------------------------------ #
rt.reps = lmer(RT_log  ~ (1|Subject_nr) + Repetitions_block, 
               data = df.reduced)
summary(rt.reps) 
fig.2a = Anova(rt.reps,
               type           = "III",
               test.statistic = "F")
fig.2a

# FIGURE 2B ---------

# --------------------------------- #
# error rate ~ stimulus repetitions #
# --------------------------------- #
acc.reps = glmer(Error_int ~ (1|Subject_nr) + Repetitions_block, 
                 data    = df.reduced,
                 family  = binomial,
                 control = glmerControl(optimizer = "Nelder_Mead"))
summary(acc.reps)
fig.2b = Anova(acc.reps ,
               type           = "III",
               test.statistic = "Chisq")
fig.2b

# FIGURE 2C ---------

# note that these analyses are done on the complete dataset

# --------------------------------------------------------------- #
# log(RT) ~ block number + condition + (block number x condition) #
# --------------------------------------------------------------- #
rt.block_cond = lmer(RT_log  ~ (1|Subject_nr) + Block_specific * Condition, 
                     data = df)

summary(rt.block_cond)
fig.2c = Anova(rt.block_cond,
               type           = "III",
               test.statistic = "F")
fig.2c

# FIGURE 2D ---------

# ------------------------------------------------------------------ #
# error rate ~ block number + condition + (block number x condition) #
# Figure 2D                                                          #
# ------------------------------------------------------------------ #
acc.block_cond = glmer(Error_int ~ (1|Subject_nr) + Block_specific * Condition, 
                       data    = df,
                       family  = binomial,
                       control = glmerControl(optimizer = "Nelder_Mead"))
summary(acc.block_cond)
fig.2d = Anova(acc.block_cond,
               type           = "III",
               test.statistic = "Chisq")
fig.2d
