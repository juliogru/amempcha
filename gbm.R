rm(list = ls(all = TRUE))

library(gbm)
library(psych)
set.seed(123)

setwd("D:\\Temp\\amazon-employee-challenge")
#load data
training = read.csv("train.csv")
test = read.csv("test.csv")




#data processing
print("Preprocessing Data")
ACTION <- as.numeric(as.character(training[,1]))
x_train <- training[,2:ncol(training)]
#standardize data
mu <- colMeans(x_train)
sigma <- SD(x_train)
x_train_scaled <- scale(x_train, center=mu, scale=sigma)
train_scaled <- data.frame(cbind(ACTION, x_train_scaled))

test_data <- data.frame(scale(test[,2:ncol(test)], center=mu, scale=sigma))
test_id <- test[,1]

#shuffle training data 
training <- train_scaled[sample.int(nrow(training)),]



#GBM model

#parameters
GBM_ITERATIONS = 5000
GBM_LEARNING_RATE = 0.01
GBM_DEPTH = 49
GBM_MINOBS = 10




x <- training[,2:ncol(train_scaled)]
y <- as.factor(as.character(training[,1]))

#cross-validation to find the optimal number of trees
gbm1 <- gbm(ACTION~. ,
			distribution = "bernoulli",
			data = training,
			n.trees = GBM_ITERATIONS,
			interaction.depth = GBM_DEPTH,
			n.minobsinnode = GBM_MINOBS,
			shrinkage = GBM_LEARNING_RATE,
			bag.fraction = 0.5,
			train.fraction = 1.0,
			cv.folds=7,
			keep.data = FALSE,
			verbose = FALSE,
			class.stratify.cv=TRUE,
			n.cores = 7)

		
		
iterations_optimal <- gbm.perf(object = gbm1 ,plot.it = TRUE,oobag.curve = TRUE,overlay = TRUE,method="cv")
print(iterations_optimal)


rm(gbm1)

#GBM Fit
x <- training[,2:ncol(train_scaled)]
y <- as.factor(as.character(training[,1]))
gbm2 <- gbm.fit(x , y
			,distribution ="bernoulli"
			,n.trees = iterations_optimal
			,shrinkage = GBM_LEARNING_RATE
			,interaction.depth = GBM_DEPTH
			,n.minobsinnode = GBM_MINOBS
			,bag.fraction = 0.5
			,nTrain = nrow(training)
			,keep.data=FALSE
			,verbose = TRUE)
		

#save submission			
ACTION <- predict.gbm(object = gbm2, newdata=test_data, n.trees=iterations_optimal, type="response")

submit_file = cbind(test_id,ACTION)
write.table(submit_file, file="gbm_4.csv",row.names=FALSE, col.names=TRUE, sep=",")

#save GBM model
#save(gbm2,file="gbm_model_1.rdata")
