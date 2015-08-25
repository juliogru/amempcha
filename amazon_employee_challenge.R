library(rpart)

setwd("D:\\Temp\\amempcha")
#setwd("P:\\Courses\\Kaggle\\amempcha")

# functions
logistic = function(x){
  1 / (1 + exp(-x))
}

binary_target = function(target, pred, alpha = 0.5, beta = 5){
  output = ifelse(target==1,1,-1)*logistic((target - ifelse(target==1,1,-1)*pred - alpha) * beta)
  return(output)
}

rescale = function(X) {
  amplitude = max(X) - min(X)
  output = rep(0,length(X))
  if (amplitude != 0) output = (X - min(X))/amplitude
  return(output)
}

reassign_random_values <- function(X, seed){
  set.seed(seed)
  values.unique = unique(X)
  nbvalues = length(values.unique)
  newvalues = sample(nbvalues, nbvalues)
  
  output = rep(NULL, length(X))
  for(i in 1:nbvalues){
    output[X == values.unique[i]] = newvalues[i]
  }
  return(output)
}

# data loading / cleaning
trainData = read.csv("train.csv")
testData  = read.csv("test.csv")
trainDataX = trainData[,2:9]
testDataX = testData[,2:9]
trainDataY = trainData$ACTION
combinedDataX = rbind(trainDataX,testDataX)

# creating higher level features
nb.features = ncol(combinedDataX)
k = nb.features + 1
for (i in seq(1,nb.features-1)) {
  for (j in seq(i+1,nb.features)) {
    cat(paste0(i,"|",j,"\n"))
    combinedDataX[,k] = paste0(combinedDataX[,i],"a",combinedDataX[,j])
    k = k + 1
  }
}

# feature selection

## Parameters
n.iterations = 10
n.trees = 1500
bag.fraction = .8
shrinkage = .1
mb = 10
mdepth = 13

# boosting rpart
Rpart.Boosted = function(trainX, trainY, testX,
                          n.trees, shrinkage,
                          bag.fraction, minbucket = 1, maxdepth = 1,
                          func){
  
  predictionTrain = rep(mean(trainY), length(trainY))
  predictionTest = rep(mean(trainY), nrow(testX))
  
  numData = nrow(trainX)
  sampsize = round(bag.fraction * numData)
  idxTrain = 1:numData
  for(j in seq(1,n.trees)){
    if(bag.fraction < 1){
      idxTrain = sample(numData, sampsize, replace = FALSE)
    }
    yy = func(trainY, predictionTrain)

    ctrl = rpart.control(minbucket = minbucket,cp = 0,
                         maxcompete = 0, maxsurrogate = 0,
                         xval = 0, maxdepth = maxdepth)
    model.fit = rpart(yy[idxTrain] ~., 
                  trainX[idxTrain,],
                  control= ctrl)

    predictionTrain = predictionTrain + shrinkage * predict(model.fit, trainX)
    predictionTest = predictionTest + shrinkage * predict(model.fit, testX)
  }
  list(training = predictionTrain, testing = predictionTest)
}

# Perform Rpart.Boosted several times and aggregate
output.prob = data.frame(matrix(NA,nrow=nrow(testDataX),ncol=n.iterations))
for (j in seq(1,n.iterations)) {
  cat("iteration #",j," date/time:",date(),"\n")
  seed = j
  combinedDataX = rbind(trainDataX,testDataX)
  for(idx in 1:ncol(combinedDataX)) {
    combinedDataX[, idx] <- reassign_random_values(combinedDataX[, idx], seed)
  }
  trainDataX = combinedDataX[1:nrow(trainDataX),]
  testDataX  = combinedDataX[-(1:nrow(trainDataX)),]
  
  set.seed(seed)
  result =  Rpart.Boosted(trainDataX, trainDataY, testDataX,
                          n.trees = n.trees, shrinkage = shrinkage, bag.fraction = bag.fraction, 
                          minbucket = mb, maxdepth = mdepth,
                          func = binary_target)
  output.prob[,j] = result$testing
}


df = data.frame(Id = testData$id, 
                 Action = rescale(rowMeans(output.prob)))
write.table(df, file="multi_boosted_rpart3.csv",row.names=FALSE, col.names=TRUE, sep=",")

