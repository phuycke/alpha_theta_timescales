##################################################
## Project: Theta and alpha power across fast and slow timescales in cognitive control
## Script purpose: main neural analyses (see 'EEG Results' in Results)
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

# FIGURE 3C ---------

# ----------------------------------------------------------------- #
# alpha power ~ repetitions + condition + (repetitions x condition) #
# ----------------------------------------------------------------- #

# remove stimulus numbers 1 and 8 to avoid double-dipping
df.reduced = df[(as.numeric(as.character(df$Repetitions_block)) > 1), ]
df.reduced = df.reduced[(as.numeric(as.character(df.reduced$Repetitions_block)) < 8), ]

alpha.reps_cond = lmer(Alpha  ~ (1|Subject_nr) + Repetitions_block * Condition, 
                       data = df.reduced)
summary(alpha.reps_cond)
fig.3c = Anova(alpha.reps_cond,
               type           = "III",
               test.statistic = "F")
fig.3c

# FIGURE 3D ---------

# ------------------------------------------------------------------- #
# alpha power ~ block number + condition + (block number x condition) #
# ------------------------------------------------------------------- #
alpha.block_cond = lmer(Alpha  ~ (1|Subject_nr) + Block_specific * Condition, 
                        data = df)
summary(alpha.block_cond)
fig.3d = Anova(alpha.block_cond,
               type           = "III",
               test.statistic = "F")
fig.3d

# FIGURE 4C ---------

# ----------------------------------------------------------------- #
# theta power ~ repetitions + condition + (repetitions x condition) #
# ----------------------------------------------------------------- #
theta.reps_cond = lmer(Theta  ~ (1|Subject_nr) + Repetitions_block * Condition, 
                       data = df)
summary(theta.reps_cond)
fig.4c = Anova(theta.reps_cond,
               type           = "III",
               test.statistic = "F")
fig.4c

# FIGURE 4D ---------

# ------------------------------------------------------------------- #
# theta power ~ block number + condition + (block number x condition) #
# ------------------------------------------------------------------- #

# remove block numbers 1 and 8 to avoid double-dipping
df.reduced = df[(as.numeric(as.character(df$Block_specific)) > 1), ]
df.reduced = df.reduced[(as.numeric(as.character(df.reduced$Block_specific)) < 8), ]

theta.block_cond = lmer(Theta  ~ (1|Subject_nr) + Block_specific * Condition, 
                        data = df.reduced)

summary(theta.block_cond)
fig.4d = Anova(theta.block_cond,
               type           = "III",
               test.statistic = "F")
fig.4d
