TrainTestSplit <- function(df, train.ratio=0.8, verbose=FALSE) {
  train.size <- floor(train.ratio * nrow(df))
  test.size <- nrow(df) - train.size
  random.idx <- sample(nrow(df), replace=FALSE)
  train.set <- df[
    head(random.idx, train.size),
  ]
  test.set <- df[
    tail(random.idx, test.size),
  ]
  
  if (verbose) {
    cat("train.size=", nrow(train.set), "test.size=", nrow(test.set))
  }
  return(list(
    "train"=train.set,
    "test"=test.set
  ))
}