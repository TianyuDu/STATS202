library(MLmetrics)
source("./util.R")

setwd("/Users/tianyudu/Documents/UToronto/2019 Summer Exchange/Stanford Summer Session/STATS202/STATS202/")
# Prepare the dataset
df.main <- read.csv("./data/train.csv", header=TRUE)
df <- TrainTestSplit(df.main, verbose=TRUE)


model.fit <- glm(
  formula=Alert ~ P1+P2+P3+P4+P5+P6+P7
    +N1+N2+N3+N4+N5+N6+N7
    +G1+G2+G3+G4+G5+G6+G7+G8+G9+G10+G11+G12+G13+G14+G15+G16,
  family=binomial(link="logit"),
  data=df$train
)

test.fit.prob <- predict(
  model.fit,
  newdata=df$test,
  type="response"
)

threshold <- 0.5
test.fit.bin <- ifelse(test.fit.prob > threshold, 1, 0)

test.accuracy <- mean(
  df$test$Alert == as.data.frame(test.fit.bin)
)

log.loss <- LogLoss(y_pred=test.fit.prob, y_true=df$test$Alert)
auc <- AUC(y_pred=test.fit.prob, y_true=df$test$Alert)
