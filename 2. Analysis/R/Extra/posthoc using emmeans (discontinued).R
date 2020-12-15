#!/usr/bin/Rscript

library(car)
library(lme4)
library(MuMIn) # rsquared
library(emmeans) # post-hoc

# SETUP + PREPARING DATA ---------
library(reticulate)
pd = import("pandas")
setwd("C:/Users/pieter/OneDrive - UGent/Projects/2019/overtraining - PILOT 3/figures/Publish/Data/Stimulus-locked/Block 1, 2, ... 8/R")
options(scipen=999) 

# Load the alpha power per repetition and put in df ---------
df = pd$read_pickle("stim_theta_blocks.pkl")

# just  check
head(df)
tail(df)

# replace spaces in col names with _
colnames(df) = gsub(" ", "_", colnames(df))


df$Subject_number   = factor(df$Subject_number)
df$Block_number     = factor(df$Block_number)
df$Condition        = factor(df$Condition)


options(contrasts=c("contr.sum", "contr.poly"))

fit     = lmer(Theta_power  ~ Block_number*Condition + (1|Subject_number), data=df) # random intercept only
#fit2    = lmer(Theta_power  ~ Block_number*Condition + (Block_number|Subject_number), data=df) # random slope + intercept
# Warning message:
#In checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv,  :
#               Model failed to converge with max|grad| = 0.00313104 (tol = 0.002, component 1)
# https://rstudio-pubs-static.s3.amazonaws.com/33653_57fc7b8e5d484c909b615d8633c01d51.html

# m marginal fixed effects - c conditional marginal effects (usually interested in marginal)
r.squaredGLMM(fit)
#r.squaredGLMM(fit2)


summary(fit) #fit of fit2, afhankelijk van wat het beste model is

Anova(fit, type = "III", test.statistic = "F") # fit of fit2, afhankelijk van wat het beste model is
# beter: https://www.rdocumentation.org/packages/lme4/versions/1.1-21/topics/bootMer

ph              = emmeans(fit, specs = pairwise  ~ Block_number:Condition, adjust = "holm") 
ph$contrasts
ph2             = emmeans(fit, specs = pairwise  ~ Block_number, adjust = "holm") 
ph2$contrasts
