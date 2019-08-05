# Analyze the treatment effects.
# Aug. 4 2019.
df <- read.csv("./Study_A_to_E_95.csv", header=TRUE, sep=",")

POLY <- 2

model <- lm(
    formula=log(PANSS_Total) ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study) + as.factor(Country),
    data=df
)
summary(model)

model <- lm(
    formula=log(P_Total) ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study),
    data=df
)
summary(model)

model <- lm(
    formula=log(N_Total) ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study),
    data=df
)
summary(model)

model <- lm(
    formula=log(G_Total) ~ poly(VisitDay, POLY)*Treatment + Treatment + as.factor(Study),
    data=df
)
summary(model)