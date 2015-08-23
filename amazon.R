library(caret)

#setwd("D:\\Temp\\amazon-employee-challenge")
setwd("P:\\Courses\\Kaggle\\amempcha")

# functions
logistic = function(x){
  1 / (1 + exp(-x))
}


# binaryTargetFN <- function(target, pred, alpha = 0.5, beta = 5){
#   ixP <- which(target == 1)
#   ixN <- which(target == 0)
#   cat("ixP:",ixP,"\n")
#   cat("ixN:",ixN,"\n")
#   target2 <- rep(0,length(pred))
#   target2[ixP] <- logistic((1 - pred[ixP] - alpha) * beta)
#   target2[ixN] <- -logistic((pred[ixN] - alpha) * beta)
#   target2
# }

binaryTargetFunc = function(target, pred, alpha = 0.5, beta = 5){
  #ixP <- which(target == 1)
  #ixN <- which(target == 0)
  #output = rep(0,length(pred))
  output = ifelse(target==1,1,-1)*logistic((target - ifelse(target==1,1,-1)*pred - alpha) * beta)
  return(output)
}

normalize <- function(a){
  deno <- max(a) - min(a)
  if(deno == 0) deno <- 1
  (a - min(a)) / deno
}

rescale = function(X) {
  amplitude = max(X) - min(X)
  output = rep(0,length(X))
  if (amplitude != 0) output = (X - min(X))/amplitude
  return(output)
}


# data loading / cleaning
trainData = read.csv("train.csv")
testData  = read.csv("test.csv")

#nrowTrain = nrow(trainData)
#combinedDataX = rbind(trainData[,2:9], testData[,2:9])
trainDataX = trainData[,2:9]
testDataX = testData[,2:9]
trainDataY = trainData$ACTION

## Parameters
nModels = 5
nt = 1500
bf = .8
sh = .1
mb = 10
mdepth = 13

# boosting rpart
BoostedRpartFitAndPredict <- function(trainX, trainY, testX,
                                      nt = 10, sh = 0.2,
                                      bf = 1, minbucket = 1, maxdepth = 1,
                                      targetFN){
  
  predTrain = rep(mean(trainY), length(trainY))
  predTest = rep(mean(trainY), nrow(testX))
  
  numData = nrow(trainX)
  sampsize = round(bf * numData)
  idxTrain = 1:numData
  for(j in seq(1,nt)){
    if(bf < 1){
      idxTrain = sample(numData, sampsize, replace = FALSE)
    }
    yy = targetFN(trainY, predTrain)
    #model <- rpart(yy[idxTrain] ~., trainX[idxTrain,],
    #               control = rpart.control(
    #                 minbucket = minbucket,
    #                 cp = 0,
    #                 maxcompete = 0,
    #                 maxsurrogate = 0,
    #                 xval = 0,
    #                 maxdepth = maxdepth))
    ctrl = rpart.control(minbucket = minbucket,cp = 0,
                         maxcompete = 0, maxsurrogate = 0,
                         xval = 0, maxdepth = maxdepth)
    model = rpart(yy[idxTrain] ~., 
                  trainX[idxTrain,],
                  control= ctrl)
                   #               control = rpart.control(
    #model = train(y = yy[idxTrain],
    #              x = trainX[idxTrain,],
    #              method = "rpart",
    #              control= ctrl)
    #print(predict(model, trainX))
    predTrain = predTrain + sh * predict(model, trainX)
    predTest = predTest + sh * predict(model, testX)
  }
  list(train = predTrain, test = predTest)
}

result =  BoostedRpartFitAndPredict(trainDataX, trainDataY, testDataX,
                                   nt = nt, sh = sh, bf = bf, minbucket = mb, maxdepth = mdepth,
                                   targetFN = binaryTargetFunc)

output.prob = result$test

df = data.frame(Id = testData$id, 
                 Action = rescale(output.prob))
write.table(df, file="boosted_rpart4.csv",row.names=FALSE, col.names=TRUE, sep=",")

