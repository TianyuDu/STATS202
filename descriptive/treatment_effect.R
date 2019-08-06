# Analyze the treatment effects.
# Aug. 4 2019.
setwd("/Users/tianyudu/Documents/UToronto/2019 Summer Exchange/Stanford Summer Session/STATS202/STATS202/descriptive")
df <- read.csv("./Study_A_to_E_95.csv", header=TRUE, sep=",")

POLY <- 2

model <- lm(
    formula=PANSS_Total ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study) + as.factor(Country) + as.factor(Study)*Treatment + as.factor(Country)*Treatment,
    data=df
)
summary(model)

model <- lm(
    formula=P_Total ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study) + as.factor(Country) + as.factor(Study)*Treatment + as.factor(Country)*Treatment,
    data=df
)
summary(model)

model <- lm(
    formula=N_Total ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study) + as.factor(Country) + as.factor(Study)*Treatment + as.factor(Country)*Treatment,
    data=df
)
summary(model)

model <- lm(
    formula=G_Total ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study) + as.factor(Country) + as.factor(Study)*Treatment + as.factor(Country)*Treatment,
    data=df
)
summary(model)
