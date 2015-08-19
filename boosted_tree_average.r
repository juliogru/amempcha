## R version 2.15.3 (2013-03-01) -- "Security Blanket"
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## foreach: 1.4.0
## doSNOW: 1.0.6
## rpart: 4.1-1

## Usage:
## Put this file into the folder which contains train.csv and test.csv
## Type the following at the command line prompt
## R --vanilla --slave < boosted_tree_average.r

## Leaderboard Public:0.90556
## It took 3269 seconds on my PC (i5_2410m 16G RAM)

t0 <- proc.time()[[3]]

library(rpart)
library(foreach)
library(doSNOW)

dTrain <- read.csv("train.csv")
dTest <- read.csv("test.csv")

numTrain <- nrow(dTrain)
target <- dTrain$ACTION

assign_random_values <- function(var, seed){
  set.seed(seed)
  varUnique <- unique(var)
  len <- length(varUnique)
  vals <- sample(len, len)
  newvar <- rep(NULL, length(var))
  for(i in 1:len){
    newvar[var == varUnique[i]] <- vals[i]
  }
  newvar
}

sigmoid <- function(x){
  1 / (1 + exp(-x))
}


binaryTargetFN <- function(target, pred, alpha = 0.5, beta = 5){
  ixP <- which(target == 1)
  ixN <- which(target == 0)
  target2 <- rep(0,length(pred))
  target2[ixP] <- sigmoid((1 - pred[ixP] - alpha) * beta)
  target2[ixN] <- -sigmoid((pred[ixN] - alpha) * beta)
  target2
}

## Fits boosted trees using user defined target function
##
## Parameters
## xTrain: the input data frame for training
## yTrain: the vector of outcome.
## xTest: the input data frame for test
## predTrain: the initial estimated values for training data
## predTrain: the initial estimated values for test data
## nt: the total number of trees to fit. (n.trees in gbm)
## sh: a shrinkage parameter applied to each tree in the expansion. (shrinkage in gbm)
## bf: the fraction of the training set observations randomly selected
##     to propose the next tree in the expansion. (bag.fraction in gbm)
## replace: a logical variable indicating whether sampling is done with replacement
## minbucket: the minimum number of observations in any terminal '<leaf>' node.
## maxdepth: Set the maximum depth of any node of the final tree, with the
##           root node counted as depth 0
## targetFN(actual, estimated, ...):
##            the target function for building each tree. It's not necessarily
##            a gradient of a loss function.
##  
BoostedRpartFitAndPredict <- function(xTrain, yTrain, xTest,
                              predTrain = NULL, predTest = NULL,
                              nt = 10, sh = 0.2,
                              bf = 1, replace = FALSE, minbucket = 1, maxdepth = 1,
                              targetFN, ...){
                              
  numData <- nrow(xTrain)
  sampsize <- round(bf * numData)
  if(is.null(predTrain)){
    predTrain <- rep(mean(yTrain), length(yTrain))
  }
  if(is.null(predTest)){
    predTest <- rep(mean(yTrain), nrow(xTest))
  }
  idxTrain <- 1:numData
  for(j in 1:nt){
    if(bf < 1){
      idxTrain <- sample(numData, sampsize, replace = replace)
    }
    yy <- targetFN(yTrain, predTrain, ...)
    model <- rpart(yy[idxTrain] ~., xTrain[idxTrain,],
                   control = rpart.control(
                     minbucket = minbucket,
                     cp = 0,
                     maxcompete = 0,
                     maxsurrogate = 0,
                     xval = 0,
                     maxdepth = maxdepth))
    predTrain <- predTrain + sh * predict(model, xTrain)
    predTest <- predTest + sh * predict(model, xTest)
  }
  list(train = predTrain, test = predTest)
}

normalize <- function(a){
  deno <- max(a) - min(a)
  if(deno == 0) deno <- 1
  (a - min(a)) / deno
}

## parallel computation
nThreads <- 4
cl <- makeCluster(nThreads, type = "SOCK")
registerDoSNOW(cl)

## parameters
nModels <- 50
nt <- 200
sh <- .1
bf <- .8
mb <- 10
mdepth <- 15

predsub <- foreach(seed = 1:nModels, .combine = cbind,
        .packages = c("rpart")) %dopar%{
  X1 <- rbind(dTrain[, 2:9], dTest[, 2:9])
  for(idx in 1:8) X1[, idx] <- assign_random_values(X1[, idx], seed)
  set.seed(seed)
  pred <- BoostedRpartFitAndPredict(X1[1:numTrain,], target,
                                    X1[-(1:numTrain), ],
                                    nt = nt, sh = sh, bf = bf, minbucket = mb, maxdepth = mdepth,
                                    targetFN = binaryTargetFN)$test
  pred
}


pred <- data.frame(id = dTest$id,
                   ACTION = normalize(rowMeans(predsub)))
write.csv(pred, "boosted_tree_average.csv", row.names = FALSE)

cat(as.integer(proc.time()[[3]] - t0), "s\n")

quit(save = "no")
