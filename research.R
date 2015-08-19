#install.packages("caret")

# rpart_1 --> 1385
# rf_1    --> 1383 / 0.63011

library(caret)
library(rpart)
library(verification)
library(randomForest)
library(gbm)

set.seed(456)
#setwd("D:\\Temp\\amazon-employee-challenge")
setwd("P:\\Courses\\Kaggle\\amempcha")

# data loading / cleaning
trainData = read.csv("train.csv")
testData  = read.csv("test.csv")
trainDataFact = trainData
testDataFact = testData

RESOURCE.values = unique(c(trainData$RESOURCE,testData$RESOURCE))
MGR_ID.values = unique(c(trainData$MGR_ID,testData$MGR_ID))
ROLE_ROLLUP_1.values = unique(c(trainData$ROLE_ROLLUP_1,testData$ROLE_ROLLUP_1))
ROLE_ROLLUP_2.values = unique(c(trainData$ROLE_ROLLUP_2,testData$ROLE_ROLLUP_2))
ROLE_DEPTNAME.values = unique(c(trainData$ROLE_DEPTNAME,testData$ROLE_DEPTNAME))
ROLE_TITLE.values = unique(c(trainData$ROLE_TITLE,testData$ROLE_TITLE))
ROLE_FAMILY_DESC.values = unique(c(trainData$ROLE_FAMILY_DESC,testData$ROLE_FAMILY_DESC))
ROLE_FAMILY.values = unique(c(trainData$ROLE_FAMILY,testData$ROLE_FAMILY))
ROLE_CODE.values = unique(c(trainData$ROLE_CODE,testData$ROLE_CODE))

trainDataFact$ACTION = factor(trainDataFact$ACTION)
trainDataFact$RESOURCE = factor(trainDataFact$RESOURCE,RESOURCE.values)
trainDataFact$MGR_ID = factor(trainDataFact$MGR_ID,MGR_ID.values)
trainDataFact$ROLE_ROLLUP_1 = factor(trainDataFact$ROLE_ROLLUP_1,ROLE_ROLLUP_1.values)
trainDataFact$ROLE_ROLLUP_2 = factor(trainDataFact$ROLE_ROLLUP_2,ROLE_ROLLUP_2.values)
trainDataFact$ROLE_DEPTNAME = factor(trainDataFact$ROLE_DEPTNAME,ROLE_DEPTNAME.values)
trainDataFact$ROLE_TITLE = factor(trainDataFact$ROLE_TITLE,ROLE_TITLE.values)
trainDataFact$ROLE_FAMILY_DESC = factor(trainDataFact$ROLE_FAMILY_DESC,ROLE_FAMILY_DESC.values)
trainDataFact$ROLE_FAMILY = factor(trainDataFact$ROLE_FAMILY,ROLE_FAMILY.values)
trainDataFact$ROLE_CODE = factor(trainDataFact$ROLE_CODE,ROLE_CODE.values)

testDataFact$RESOURCE = factor(testDataFact$RESOURCE,RESOURCE.values)
testDataFact$MGR_ID = factor(testDataFact$MGR_ID,MGR_ID.values)
testDataFact$ROLE_ROLLUP_1 = factor(testDataFact$ROLE_ROLLUP_1,ROLE_ROLLUP_1.values)
testDataFact$ROLE_ROLLUP_2 = factor(testDataFact$ROLE_ROLLUP_2,ROLE_ROLLUP_2.values)
testDataFact$ROLE_DEPTNAME = factor(testDataFact$ROLE_DEPTNAME,ROLE_DEPTNAME.values)
testDataFact$ROLE_TITLE = factor(testDataFact$ROLE_TITLE,ROLE_TITLE.values)
testDataFact$ROLE_FAMILY_DESC = factor(testDataFact$ROLE_FAMILY_DESC,ROLE_FAMILY_DESC.values)
testDataFact$ROLE_FAMILY = factor(testDataFact$ROLE_FAMILY,ROLE_FAMILY.values)
testDataFact$ROLE_CODE = factor(testDataFact$ROLE_CODE,ROLE_CODE.values)

trainData$ACTION = factor(trainData$ACTION)

# data split into training set / testing set
inTrain = createDataPartition(y = trainDataFact$ACTION,
                              p = 0.60,
                              list = FALSE)

trainingFact = trainDataFact[inTrain,]
testingFact = trainDataFact[-inTrain,]

training = trainData[inTrain,]
testing = trainData[-inTrain,]

# rpart
ctrl = rpart.control(minsplit = 20, minbucket = 7, maxdepth = 10, complexity = 0)
fit1 = rpart(ACTION ~ .,method="class", data=trainingFact,
             control = ctrl)

pred_action_1 = predict(fit1, newdata = testingFact, type = "class")
roc.area(as.numeric(as.character(testingFact$ACTION)), as.numeric(as.character(pred_action_1)))$A

# rf
fit2 = randomForest(ACTION ~ ., data=training, 
                    importance=TRUE, na.action=na.omit)

pred_action_2 = predict(fit2, newdata = testing)
roc.area(as.numeric(as.character(testing$ACTION)), as.numeric(as.character(pred_action_2)))$A

# gbm
fit3 = gbm(ACTION ~ ., data = training,
           n.trees = 10,
           distribution = "bernoulli",
           cv.folds=10,
           n.cores=4)

pred_action_3 = predict(fit3, newdata = testing, n.trees = 100)
roc.area(as.numeric(as.character(testing$ACTION)), as.numeric(as.character(pred_action_3)))$A

# caret / eXtreme Gradient Boosting

fit4 = train(ACTION ~ .,data = training,
             method = "gbm")

pred_action_4 = predict(fit4, newdata = testing, type = "prob")

roc.area(as.numeric(as.character(testing$ACTION)), as.numeric(as.character(pred_action_3)))$A

# submission preparation
testId = testData[,"id"]

output_1 = cbind(testId,as.numeric(as.character(predict(fit1, newdata = testDataFact, type = "class"))))
colnames(output_1) = c("Id","Action")
write.table(output_1, file="rpart_1.csv",row.names=FALSE, col.names=TRUE, sep=",")

output_2 = cbind(testId,as.numeric(as.character(predict(fit2, newdata = testData))))
colnames(output_2) = c("Id","Action")
write.table(output_2, file="rf_1.csv",row.names=FALSE, col.names=TRUE, sep=",")
