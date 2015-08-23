## R version 2.15.3 (2013-03-01) -- "Security Blanket"
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## foreach: 1.4.0
## doSNOW: 1.0.6
## rpart: 4.1-1

## Usage:
## Put this file into the folder which contains train.csv and test.csv
## Type the following at the command line prompt
## R --vanilla --slave < boosted_tree_average2.r

## Leaderboard Public:0.91588 Private:0.91392
## 6.6 hours  PC2 (AMD phenomII945, 8G)


t0 <- proc.time()[[3]]

library(rpart)
library(foreach)
#library(doSNOW)
library(doParallel)

setwd("P:\\Courses\\Kaggle\\amempcha")
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
##            a *negative* gradient of a loss function.
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
    print(predict(model, xTrain))
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

# Miroslaw's idea
group_data <- function(data, degree = 2){
  m <- ncol(data)
  indicies <- combn(1:m, degree)
  dataStr <- apply(indicies, 2, function(s) apply(data[,s], 1, function(x) paste0(x, collapse = "a")))
  dataStr
}

# All infrequent categories into one category
FactorVar <- function(v, lb = 20){
  tbl <- sort(table(v), decreasing = TRUE)
  len <- length(tbl)
  freqEnough <- which(tbl > lb)
  if(length(freqEnough) == 0) return(NULL) # Drop this variable
  k <- max(freqEnough)
  if(k < len){
    vals <- names(tbl[1:k])
    v[! v %in% vals] <- "other"
  }
  factor(v)
}

combn2str <- function(x, m){
  apply(combn(x, m), 2, function(x) paste0(x, collapse = ""))
}

## Feature extraction
# original features + the second level group data 
dAll <- rbind(dTrain[,2:9], dTest[,2:9])  
X1 <- dAll
X2 <- group_data(dAll, 2)

# Remove redundant features
Xall <- data.frame(X1, X2[,-27], stringsAsFactors = FALSE)


# All infrequent categories into one category
# I only tried threshold 1.
XallF <- FactorVar(Xall[,1], 1)
for(j in 2:ncol(Xall)){
  x <- FactorVar(Xall[,j], 1)
  XallF <- data.frame(XallF, x)
}

# Factor to integer
XallN <- data.frame(lapply(XallF, as.integer))


## Parallel computation
nThreads <- 4
#cl <- makeCluster(nThreads, type = "SOCK")
#registerDoSNOW(cl)
cl <- makeCluster(nThreads)
registerDoParallel(cl)

## Parameters
nModels <- 5
nt <- 1500
bf <- .8
sh <- .1
mb <- 10
mdepth <- 13


## 
pred2sub <- foreach(seed = 1:nModels, .combine = cbind,
        .packages = c("rpart")) %dopar%{
  X <- XallN
  for(idx in 1:ncol(XallN)) X[, idx] <- assign_random_values(XallN[, idx], seed)
  set.seed(seed)
  pred <- BoostedRpartFitAndPredict(X[1:numTrain,10], target[10],
                                    X[-(1:numTrain) , 10],
                                    nt = nt, sh = sh, bf = bf, minbucket = mb, maxdepth = mdepth,
                                    targetFN = binaryTargetFN)$test
  pred
}

X <- XallN
pred <- BoostedRpartFitAndPredict(X[1:numTrain,1:10], target[1:10],
                                  X[-(1:numTrain) , 1:10],
                                  nt = nt, sh = sh, bf = bf, minbucket = mb, maxdepth = mdepth,
                                  targetFN = binaryTargetFN)$test

# only with level 1 features
pred2sub <- foreach(seed = 1:nModels, .combine = cbind, .packages = c("rpart")) %dopar%{
  X <- X1
  for(idx in 1:ncol(X1)) X[, idx] <- assign_random_values(X1[, idx], seed)
  set.seed(seed)
  pred <- BoostedRpartFitAndPredict(X[1:numTrain,], target,
                                    X[-(1:numTrain) , ],
                                    nt = nt, sh = sh, bf = bf, minbucket = mb, maxdepth = mdepth,
                                    targetFN = binaryTargetFN)$test
  pred
}

X <- X1
pred <-  BoostedRpartFitAndPredict(X[1:numTrain,], target,
                          X[-(1:numTrain) , ],
                          nt = nt, sh = sh, bf = bf, minbucket = mb, maxdepth = mdepth,
                          targetFN = binaryTargetFN)$test


#pred <- data.frame(id = dTest$id,
#                   ACTION = normalize(rowMeans(pred2sub)),
#                   )

pred2sub = cbind(c(pred))

pred <- data.frame(id = dTest$id,
                   ACTION = normalize(rowMeans(pred2sub)))

write.csv(pred, "boosted_rpart1.csv", row.names = FALSE)

cat(as.integer(proc.time()[[3]] - t0), 