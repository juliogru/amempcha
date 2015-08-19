#install.packages("caret")
library(caret)
library(rpart)
library(verification)

set.seed(456)
setwd("D:\\Temp\\amazon-employee-challenge")

# data loading / cleaning
trainData = read.csv("train.csv")
testData  = read.csv("test.csv")

RESOURCE.values = unique(c(trainData$RESOURCE,testData$RESOURCE))
MGR_ID.values = unique(c(trainData$MGR_ID,testData$MGR_ID))
ROLE_ROLLUP_1.values = unique(c(trainData$ROLE_ROLLUP_1,testData$ROLE_ROLLUP_1))
ROLE_ROLLUP_2.values = unique(c(trainData$ROLE_ROLLUP_2,testData$ROLE_ROLLUP_2))
ROLE_DEPTNAME.values = unique(c(trainData$ROLE_DEPTNAME,testData$ROLE_DEPTNAME))
ROLE_TITLE.values = unique(c(trainData$ROLE_TITLE,testData$ROLE_TITLE))
ROLE_FAMILY_DESC.values = unique(c(trainData$ROLE_FAMILY_DESC,testData$ROLE_FAMILY_DESC))
ROLE_FAMILY.values = unique(c(trainData$ROLE_FAMILY,testData$ROLE_FAMILY))
ROLE_CODE.values = unique(c(trainData$ROLE_CODE,testData$ROLE_CODE))

trainData$ACTION = factor(trainData$ACTION)
trainData$RESOURCE = factor(trainData$RESOURCE,RESOURCE.values)
trainData$MGR_ID = factor(trainData$MGR_ID,MGR_ID.values)
trainData$ROLE_ROLLUP_1 = factor(trainData$ROLE_ROLLUP_1,ROLE_ROLLUP_1.values)
trainData$ROLE_ROLLUP_2 = factor(trainData$ROLE_ROLLUP_2,ROLE_ROLLUP_2.values)
trainData$ROLE_DEPTNAME = factor(trainData$ROLE_DEPTNAME,ROLE_DEPTNAME.values)
trainData$ROLE_TITLE = factor(trainData$ROLE_TITLE,ROLE_TITLE.values)
trainData$ROLE_FAMILY_DESC = factor(trainData$ROLE_FAMILY_DESC,ROLE_FAMILY_DESC.values)
trainData$ROLE_FAMILY = factor(trainData$ROLE_FAMILY,ROLE_FAMILY.values)
trainData$ROLE_CODE = factor(trainData$ROLE_CODE,ROLE_CODE.values)

testData$RESOURCE = factor(testData$RESOURCE,RESOURCE.values)
testData$MGR_ID = factor(testData$MGR_ID,MGR_ID.values)
testData$ROLE_ROLLUP_1 = factor(testData$ROLE_ROLLUP_1,ROLE_ROLLUP_1.values)
testData$ROLE_ROLLUP_2 = factor(testData$ROLE_ROLLUP_2,ROLE_ROLLUP_2.values)
testData$ROLE_DEPTNAME = factor(testData$ROLE_DEPTNAME,ROLE_DEPTNAME.values)
testData$ROLE_TITLE = factor(testData$ROLE_TITLE,ROLE_TITLE.values)
testData$ROLE_FAMILY_DESC = factor(testData$ROLE_FAMILY_DESC,ROLE_FAMILY_DESC.values)
testData$ROLE_FAMILY = factor(testData$ROLE_FAMILY,ROLE_FAMILY.values)
testData$ROLE_CODE = factor(testData$ROLE_CODE,ROLE_CODE.values)

# data split into training set / testing set
inTrain = createDataPartition(y = trainData$ACTION,
                              p = 0.60,
                              list = FALSE)

training = trainData[inTrain,]
testing = trainData[-inTrain,]

# rpart
ctrl = rpart.control(minsplit = 20, minbucket = 7, maxdepth = 7, complexity = 0)
fit1 = rpart(ACTION ~ .,method="class", data=training,
             control = ctrl)

pred_action_1 = predict(fit1, newdata = testing, type = "class")
auc(as.numeric(as.character(testing$ACTION)), as.numeric(as.character(pred_action_1)))


# submission preparation
testId = testData[,"id"]

output_1 = cbind(testId,as.numeric(as.character(predict(fit1, newdata = testData, type = "class"))))
colnames(output_1) = c("Id","Action")
write.table(output_1, file="rpart_1.csv",row.names=FALSE, col.names=TRUE, sep=",")

