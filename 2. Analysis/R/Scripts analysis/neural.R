#!/usr/bin/Rscript

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

# SLOW TIMESCALE ---------

# remove the first and the last blocks to avoid double-dipping
df.reduced = df[(as.numeric(as.character(df$Block_specific)) > 1), ]
df.reduced = df.reduced[(as.numeric(as.character(df.reduced$Block_specific)) < 8), ]

# ------------------------------------------------------------------- #
# theta power ~ block number + condition + (block number x condition) #
# ------------------------------------------------------------------- #
theta.block_cond = lmer(Theta  ~ (1|Subject_nr) + Block_specific * Condition, 
                        data = df.reduced)

summary(theta.block_cond)
aov1 = Anova(theta.block_cond,
             type           = "III",
             test.statistic = "F")
aov1

# FAST TIMESCALE ---------

# remove repetitions larger than 9
df.reduced = df[(as.numeric(as.character(df$Repetitions_overall)) <= 8), ]

# make a subset for when repetitions <= 8
df.reduced = df[(as.numeric(as.character(df$Repetitions_block)) > 1), ]
df.reduced = df.reduced[(as.numeric(as.character(df.reduced$Repetitions_block)) < 8), ]

# ------------------------- #
# alpha power ~ repetitions #
# ------------------------- #
alpha.reps = lmer(Alpha ~ (1|Subject_nr) + Repetitions_block, 
                  data = df.reduced)
summary(alpha.reps)
aov2 = Anova(alpha.reps,
             type           = "III",
             test.statistic = "F")
aov2

# ------------------------ #
# beta power ~ repetitions #
# ------------------------ #
beta.reps = lmer(Beta ~ (1|Subject_nr) + Repetitions_block, 
                data = df.reduced)
summary(beta.reps)
aov3 = Anova(beta.reps,
             type           = "III",
             test.statistic = "F")
aov3

# ADDITIONAL CHECKS ---------

# ------------------------- #
# theta power ~ repetitions #
# ------------------------- #

# remove repetitions larger than 9
df.reduced = df[(as.numeric(as.character(df$Repetitions_overall)) < 9), ]

theta.reps = lmer(Theta ~ (1|Subject_nr) + Repetitions_block, 
                  data = df.reduced)
summary(theta.reps)
aov4 = Anova(theta.reps,
             type           = "III",
             test.statistic = "F")
aov4

# ----------------------------------------------------------------- #
# alpha power ~ repetitions + condition + (repetitions x condition) #
# ----------------------------------------------------------------- #
# make a subset for when repetitions <= 8
df.reduced = df[(as.numeric(as.character(df$Repetitions_block)) > 1), ]
df.reduced = df.reduced[(as.numeric(as.character(df.reduced$Repetitions_block)) < 8), ]

alpha.reps_cond = lmer(Alpha  ~ (1|Subject_nr) + Repetitions_block * Condition, 
                       data = df.reduced)
summary(alpha.reps_cond)
aov5 = Anova(alpha.reps_cond,
             type           = "III",
             test.statistic = "F")
aov5

# ------------------------------------------------------------------- #
# alpha power ~ block number + condition + (block number x condition) #
# ------------------------------------------------------------------- #
alpha.block_cond = lmer(Alpha  ~ (1|Subject_nr) + Block_specific * Condition, 
                        data = df)
summary(alpha.block_cond)
aov6 = Anova(alpha.block_cond,
             type           = "III",
             test.statistic = "F")
aov6

# ----------------------------------------------------------------- #
# theta power ~ repetitions + condition + (repetitions x condition) #
# ----------------------------------------------------------------- #
theta.reps_cond = lmer(Theta  ~ (1|Subject_nr) + Repetitions_block * Condition, 
                       data = df)
summary(theta.reps_cond)
aov8 = Anova(theta.reps_cond,
             type           = "III",
             test.statistic = "F")
aov8

# ------------------------------------------------------------------- #
# beta power ~ block number + condition + (block number x condition) #
# ------------------------------------------------------------------- #
beta.block_cond = lmer(Beta  ~ (1|Subject_nr) + Block_specific * Condition, 
                       data = df)
summary(beta.block_cond)
aov7 = Anova(beta.block_cond,
             type           = "III",
             test.statistic = "F")
aov7


# ------------ #
# New analysis #
# ------------ #

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

# analysis for the fast timescale
analysis.1 = lmer(Power  ~ (1|Subject_nr) + Repetitions_block * Condition * Power_dummy, 
                  data = df_powersplit)
summary(analysis.1)
aov9 = Anova(analysis.1,
             type           = "III",
             test.statistic = "F")
aov9

# analysis for the slow timescale
analysis.2 = lmer(Power  ~ (1|Subject_nr) + Block_specific * Condition * Power_dummy, 
                  data = df_powersplit)
summary(analysis.2)
aov10 = Anova(analysis.2,
             type           = "III",
             test.statistic = "F")
aov10



