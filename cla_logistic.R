library(MLmetrics)
source("./util.R")

# Prepare the dataset
df.main <- read.csv("./data/train.csv", header=TRUE)
df <- TrainTestSplit(df.main, verbose=TRUE)

model.fit <- glm(
  formula=Pass ~ .-X-Study-Country+as.factor(Country)
  -PatientID-SiteID-RaterID-AssessmentiD-TxGroup+as.factor(TxGroup)-LeadStatus,
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
  df$test$Pass == as.data.frame(test.fit.bin)
)

log.loss <- LogLoss(y_pred=test.fit.prob, y_true=df$test$Pass)
