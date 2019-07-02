TrainTestSplit <- function(df, train.ratio=0.8, verbose=FALSE) {
  train.size <- floor(train.ratio * nrow(df))
  test.size <- nrow(df) - train.size
  if (verbose) {
    cat("train.size=", train.size, "test.size=", test.size)
  }
  random.idx <- sample(nrow(df), replace=FALSE)
  train.set <- df[1:train.size,]
  test.set <- df[train.size+1:nrow(df),]
  return(list(
    "train"=train.set,
    "test"=test.set
  ))
}