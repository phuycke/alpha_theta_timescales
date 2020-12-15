#!/usr/bin/Rscript

# ----
# Define the summary function used for plotting
# ---------------------------------------------

summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This is does the summary; it's not easy to understand...
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun= function(xx, col, na.rm) {
                   c( N    = length2(xx[,col], na.rm=na.rm),
                      mean = mean   (xx[,col], na.rm=na.rm),
                      sd   = sd     (xx[,col], na.rm=na.rm)
                   )
                 },
                 measurevar,
                 na.rm
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean"=measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}

scale_f <- function(x) sprintf("%.2f", x)
scale_d <- function(x) sprintf("%.2f", as.numeric(as.character(x)))

# ----
# Libraries
# ---------

library(reticulate)
library(tidyverse)
library(ggpubr)
library(rstatix)
library(car)
library(lme4)
library(plyr)
library(ggplot2)

# ----
# Prepare + read the data
# -----------------------

# general settings
setwd("C:/Users/pieter/OneDrive - UGent/Projects/2019/overtraining - PILOT 3/code/Analysis/Behavioral/General/Data")
options(scipen = 999)
options(contrasts = c("contr.sum", "contr.poly"))

# Load the alpha power per repetition and put in df
pd = import("pandas")
df = pd$read_pickle("behavioral_clean.pkl")

# replace spaces in col names with _
colnames(df) = gsub(" ", "_", colnames(df))

# ----
# log(RT) ~ (block nr) + condition + ((block nr) * condition)
# -------------------------------------------------------------
  
# Prepare the data for analysis
df$Log_Reaction_times = log10(df$Reaction_times)                            # log scale RTs
mapvalues(df$Block_ID,                                                      # rescale the condition identifier
          c(0, 1), 
          c(-.5, .5))                                 
df$Block_number_specific = scale(df$Block_number_specific, scale = FALSE)   # mean centre the block numbers

df$Block_number_specific = factor(df$Block_number_specific)                 # see as factor
df$Subject_number        = factor(df$Subject_number)                        # idem
df$Block_ID              = factor(df$Block_ID)                              # idem

# Check for empty cells before analysis
xtabs(~Block_number_specific + Block_ID, df)

# log(RT) ~ (block nr) + condition + ((block nr) * condition)
rt.slow = lmer(Log_Reaction_times  ~ (1|Subject_number) + Block_number_specific * Block_ID, 
               data = df)
summary(rt.slow)
aov.rt_slow = Anova(rt.slow,
                    type           = "III",
                    test.statistic = "F")

# ----
# FOLLOW-UP: RT across conditions
# -------------------------------

novel     = df[df$Block_ID == 0, ]
repeating = df[df$Block_ID == 1, ]

# test if RT is significantly larger in block 1 vs. block 8 for the novel condition
pairwise.t.test(novel$Log_Reaction_times, 
                novel$Block_number_specific, 
                p.adj = "BH",
                alternative = "less")

# test if RT is significantly larger in block 1 vs. block 8 for the repeating condition
pairwise.t.test(repeating$Log_Reaction_times, 
                repeating$Block_number_specific, 
                p.adj = "BH",
                alternative = "less")


# ----
# PLOT: log(RT) ~ (block nr) * condition
# -------------------------------------

tgc <- summarySE(df, 
                 measurevar = "Log_Reaction_times", 
                 groupvars  = c("Block_number_specific","Block_ID"))
pd <- position_dodge(0.1)                                          # move them .05 to the left and right
ggplot(tgc, 
       aes(x      = Block_number_specific, 
           y      = Log_Reaction_times, 
           colour = Block_ID, 
           group  = Block_ID)) +
  geom_errorbar(aes(ymin = Log_Reaction_times - ci, 
                    ymax = Log_Reaction_times + ci), 
                colour   = "black", 
                width    = .1, 
                position = pd) +
  geom_line(position = pd) +
  geom_point(position = pd, 
             size     = 3, 
             shape    = 21, 
             fill     = "white") +                                 # 21 is filled circle
  xlab("Block number (mean centered)") +
  ylab("Reaction times (log scaled)")  +
  scale_colour_hue(name   = "Condition",                           # Legend label, use darker colors
                   breaks = c(0, 1),
                   labels = c("Repeating", "Not repeating"),
                   l      = 40) +                                  # Use darker colors, lightness = 40
  ggtitle("log(RT) ~ Block number * condition") +
  scale_y_continuous(labels = scale_f) +
  scale_x_discrete(labels = scale_d)   +                           # Set tick every 4
  theme_bw()                           +
  theme(legend.justification = c(0,0),
        legend.position      = c(0,0),
        legend.direction     = "vertical",
        legend.key.size      = unit(1.5, "cm"),
        legend.key.width     = unit(0.5,"cm"))                     # Position legend in bottom right

# ----
# accuracy ~ (block nr) + condition + ((block nr) * condition)
# -------------------------------------------------------------

# make an extra variable coding for errors made
df$error = df$Correct
df$error = abs(df$err - 1)

acc.slow = glmer(error ~ (1|Subject_number) + Block_number_specific * Block_ID,
                 data    = df,
                 family  = binomial,
                 control = glmerControl(optimizer ="Nelder_Mead"))
summary(acc.slow)
aov.acc_slow = Anova(acc.slow,
                     type           = "III",
                     test.statistic = "Chisq")

# ----
# FOLLOW-UP: accuracy across conditions
# -------------------------------

novel     = df[df$Block_ID == 0, ]
repeating = df[df$Block_ID == 1, ]

# test if RT is significantly larger in block 1 vs. block 8 for the novel condition
pairwise.t.test(novel$error, 
                novel$Block_number_specific, 
                p.adj = "BH",
                alternative = "less")

# test if RT is significantly larger in block 1 vs. block 8 for the repeating condition
pairwise.t.test(repeating$error, 
                repeating$Block_number_specific, 
                p.adj = "BH",
                alternative = "less")

# ----
# log(RT) ~ (stimulus nr)
# ---------------------

# Only with data where stimulus number < 9
df_subset = df[df$Overall_repetitions < 9, ]

# scale repetition count, and see it as a factor
df_subset$Overall_repetitions = scale(df_subset$Overall_repetitions,
                                      scale = FALSE)
df_subset$Overall_repetitions = factor(df_subset$Overall_repetitions)

# Check for empty cells before analysis
xtabs(~Overall_repetitions, df_subset)

# fit the model
rt.fast = lmer(Log_Reaction_times  ~ (1|Subject_number) + Overall_repetitions, 
               data = df_subset)
summary(rt.fast)
aov.rt_fast = Anova(rt.fast, 
                    type           = "III", 
                    test.statistic = "F")

# ----
# accuracy ~ (stimulus nr)
# ---------------------

acc.fast = glmer(Correct ~ (1|Subject_number) + Overall_repetitions,
                 data    = df_subset,
                 family  = binomial,
                 control = glmerControl(optimizer ="Nelder_Mead"))
summary(acc.fast)
aov.acc_fast = Anova(acc.fast,
                     type           = "III",
                     test.statistic = "Chisq")
