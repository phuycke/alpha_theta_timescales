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

# --------------------------------------------------------------- #
# log(RT) ~ block number + condition + (block number x condition) #
# --------------------------------------------------------------- #
rt.block_cond = lmer(RT_log  ~ (1|Subject_nr) + Block_specific * Condition, 
                     data = df)

summary(rt.block_cond)
aov1 = Anova(rt.block_cond,
             type           = "III",
             test.statistic = "F")
aov1

# ------------------------------------------------------------------ #
# error rate ~ block number + condition + (block number x condition) #
# ------------------------------------------------------------------ #
acc.block_cond = glmer(Error_int ~ (1|Subject_nr) + Block_specific * Condition, 
                       data    = df,
                       family  = binomial,
                       control = glmerControl(optimizer = "Nelder_Mead"))
summary(acc.block_cond)
aov2 = Anova(acc.block_cond,
             type           = "III",
             test.statistic = "Chisq")
aov2

# FAST TIMESCALE ---------

# make a subset for when repetitions <= 8
df.reduced = df[as.numeric(as.character(df$Repetitions_overall)) <= 8, ]

# --------------------- #
# log(RT) ~ repetitions #
# --------------------- #
rt.reps = lmer(RT_log  ~ (1|Subject_nr) + Repetitions_block, 
               data = df.reduced)
summary(rt.reps) 
aov3 = Anova(rt.reps,
             type           = "III",
             test.statistic = "F")
aov3

# ------------------------ #
# error rate ~ repetitions #
# ------------------------ #
acc.reps = glmer(Error_int ~ (1|Subject_nr) + Repetitions_block, 
                 data    = df.reduced,
                 family  = binomial,
                 control = glmerControl(optimizer = "Nelder_Mead"))
summary(acc.reps)
aov4 = Anova(acc.reps ,
             type           = "III",
             test.statistic = "Chisq")
aov4
