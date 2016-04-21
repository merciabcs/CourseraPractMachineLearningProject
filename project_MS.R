library(caret)
library(randomForest)
library(rpart)
library(rattle)

find

#download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
#download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
result_data <- read.csv("pml-testing.csv", sep = ",", na.strings = c("", "NA"))
train_data <- read.csv("pml-training.csv", sep = ",", na.strings = c("", "NA"))

# Remove columns full of NAs.
features <- names(result_data[,colSums(is.na(result_data)) == 0])[8:59]
# Only use features used in submit cases.
train_data <- train_data[,c(features,"classe")]
result_data <- result_data[,c(features,"problem_id")]

# bootstrap
set.seed(5)
inTrain = createDataPartition(train_data$classe, p = 0.75, list = F)
training = train_data[inTrain,]
testing = train_data[-inTrain,]

# feature selection
outcome = which(names(training) == "classe")
highCorrCols = findCorrelation(abs(cor(training[,-outcome])),0.90)
highCorrFeatures = names(training)[highCorrCols]
highCorrFeatures
training = training[,-highCorrCols]
outcome = which(names(training) == "classe")

#feature importance
fsRF = randomForest(training[,-outcome], training[,outcome], importance = T)
rfImp = data.frame(fsRF$importance)
impFeatures = order(-rfImp$MeanDecreaseGini)
inImp = createDataPartition(train_data$classe, p = 0.05, list = F)
featurePlot(training[inImp,impFeatures[1:4]],training$classe[inImp], plot = "pairs")

#The most important features are:
#pitch_belt
#yaw_belt
#total_accel_belt
#gyros_belt_x

# train RF and k-nearest neighbors
ctrlKNN = trainControl(method = "adaptive_cv")
modelKNN = train(classe ~ ., training, method = "knn", trControl = ctrlKNN)
ctrlRF = trainControl(method = "oob")
modelRF = train(classe ~ ., training, method = "rf", ntree = 200, trControl = ctrlRF)
modelDT= train(classe ~ ., training, method="rpart")


resultsKNN = data.frame(modelKNN$results)
resultsRF = data.frame(modelRF$results)
resultsDT = data.frame(modelDT$results)

fitKNN = predict(modelKNN, testing)
fitRF = predict(modelRF, testing)
fitDT = predict(modelDT, testing)

confusionMatrix(fitKNN, testing$classe)$overall
confusionMatrix(fitRF, testing$classe)$overall
confusionMatrix(fitDT, testing$classe)$overall
fancyRpartPlot(modelDT$finalModel)

submit = predict(modelRF, result_data)
answer = data.frame(submit)
answer
