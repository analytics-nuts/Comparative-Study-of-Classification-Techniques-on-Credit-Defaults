#Fitting Support Vector Machines##

# Data_for_SVM
data.svm = cbind.data.frame(quanti.norm,quali,target) # SVM accepts factor variables
data.svm = setDT(data.svm)

# Splitting the data into 80:20 ratio
set.seed(666)
ind=sample(nrow(data.svm),nrow(data.svm)*0.8,replace = F)
train.svm=data.svm[ind,]
test.svm=data.svm[-ind,]

## Parallel Computation for svm
set.seed(112233)
library(parallel) 
# Calculate the number of cores
no_cores <- detectCores()-1

library(doParallel)
# create the cluster for caret to use
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)

## Tuning Hyperparameters for SVM
SVM_Radial_Fit = train(target~.,train.svm, method = "svmRadial",verbose = F,
                       trControl = trainControl(method = "cv",
                                                number = 10,allowParallel = T))

stopCluster(cl)
registerDoSEQ()

SVM_Radial_Fit$bestTune

# Model Fitting
set.seed(666)
model.svm = svm(target ~ .,data=train.svm,cost = 1, gamma = 0.1885286,
                type="C-classification",kernel="radial",
                probability = T)

# For test set
pred.svm = predict(model.svm,newdata = test.svm[,-24],probability = T)

pred.svm.prob = as.data.frame(attributes(pred.svm)$probabilities)
names(pred.svm.prob) = c("no","yes")
conf5 = table(predicted = pred.svm,true = test.svm$target)
err2[5] = 1 - sum(diag(conf5))/sum(conf5)

confusionMatrix(conf5)

# For training set
pred.svm.train = predict(model.svm,newdata = train.svm[,-24],probability = T)
pred.svm.train.prob = as.data.frame(attributes(pred.svm.train)$probabilities)
names(pred.svm.train.prob) = c("no","yes")
conf5.train = table(predicted = pred.svm.train,true = train.svm$target)
err1[5] = 1 - sum(diag(conf5))/sum(conf5)

confusionMatrix(conf5.train)

# ROC curve and AUC value
par(mfrow=c(1,2))
roc(train.svm$target,pred.svm.train.prob$yes,plot=T,
    col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",
    ylab="True Positive percentage",
    lwd=1,main="Train Set")

roc(test.svm$target,pred.svm.prob$yes,plot=T,
    col="#404080",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",
    ylab="True Positive percentage",
    lwd=1,main="Test Set")
par(mfrow=c(1,1))
auc1[5] = auc(train.svm$target,pred.svm.train.prob[,2])
auc2[5] = auc(test.svm$target,pred.svm.prob[,2])

#Cumulative gain chart for test set

area_mod=performance(prediction(sort(pred.svm.prob$yes,decreasing=F),
                                test.svm$target[order(pred.svm.prob$yes,decreasing=F)])
                     ,measure = "auc")@y.values[[1]] #Area under curve

x=unlist(slot(performance(prediction(sort(pred.svm.prob$yes,decreasing=F),
                                     test.svm$target[order(pred.svm.prob$yes,decreasing=F)])
                          ,"tpr","rpp"),'x.values'))
y=unlist(slot(performance(prediction(sort(pred.svm.prob$yes,decreasing=F),
                                     test.svm$target[order(pred.svm.prob$yes,decreasing=F)])
                          ,"tpr","rpp"),'y.values'))
par(mfrow=c(1,1))
par(pty="m")
plot(x,y,type="l",col="green",lwd=4,main="Cumulative gain chart for SVM model",
     font.main=2,xlab="Rate of Positive Prediction",
     ylab="Cumulative proportion of target")
abline(a=0,b=1,col="red")
text(0.8,0.2,"Area under model\n curve is 0.669",font = 2,col="steelblue4")
text(0.1,0.85,"best line->",font = 2,col="steelblue4")
text(0.8,0.69,"<-base line",font = 2,col="steelblue4")
lines(x=c(0,0.2166667,1),y=c(0,1,1),lwd=2,col="blue")