## Artifical Neural Network Classifier ##

data.ann =cbind(quanti.norm, quali.dummy, target)
setDF(data.ann) #Converting into data.table

#Test train split in 80:20 ratio
set.seed(666) #For Reproducibility

ind=sample(nrow(data.ann),nrow(data.ann)*0.8,replace = F)
train.ann=data.ann[ind,]
test.ann=data.ann[-ind,]

#Parallel computation for ann
set.seed(112233)

library(parallel)
detectCores()
no_cores <- detectCores()-1

library(doParallel)
#Create the cluster for caret to use
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)



## Tuning Hyperparameters for ANN
set.seed(666)
param = trainControl(method = "cv",number = 5 , allowParallel = T,classProbs = T,summaryFunction = twoClassSummary,search = "grid")
ann.fit = train(recode_factor( target,'0'="no",'1'="yes")~., train.ann,
                method = "nnet",
                trControl = param,
                metric = "ROC",
                trace = FALSE,
                maxit = 200)

# Stop parallel Computation
stopCluster(cl)
registerDoSEQ()

#getModelInfo()$nnet

ann.fit$bestTune
plot(ann.fit)

## Model Fitting based on the Grid Search
set.seed(666)
model.ann = nnet(target~., data = train.ann, size =3, decay = 0.1,maxit=200)

## For Test Set
pred.ann.prob = predict(model.ann, newdata= test.ann[,-88])
pred.ann = ifelse(pred.ann.prob > 0.5, "1", "0" )
conf6 = table(predict= pred.ann , true = test.ann$target)
err2[6] = (conf6[1,2]+conf6[2,1])/sum(conf6)


#For Training Set

pred.ann.train.prob = predict(model.ann , newdata = train.ann[,-88])
pred.ann.train = ifelse(pred.ann.train.prob > 0.5, "1", "0" )
conf6.train = table(predicted = pred.ann.train, true = train.ann$target)
err1[6] = 1-sum(diag(conf6.train))/sum(conf6.train)

confusionMatrix(conf6.train)

#ROC curve and AUC value
par(mfrow=c(1,2))
par(pty="s")

roc(train.ann$target , pred.ann.train.prob , plot = T,col = "#69b3a2", print.auc = T, legacy.axes = TRUE , percent = T,
    xlab = "False Positive percentage", ylab = "True Positive percentage",lwd = 5, main = "Train Set")

roc(test.ann$target , pred.ann.prob , plot = T,col = "navyblue", print.auc = T, legacy.axes = TRUE , percent = T,
    xlab = "False Positive percentage", ylab = "True Positive percentage",lwd = 5, main = "Test Set")

auc1[6] = auc(train.ann$target , pred.ann.train.prob)
auc2[6] = auc(test.ann$target , pred.ann.prob)

#Cumulative gain chart for test set

area_mod=performance(prediction(sort(pred.ann.prob,decreasing=F),test.ann$target[order(pred.ann.prob,decreasing=F)])
                     ,measure = "auc")@y.values[[1]] #Area under curve

x=unlist(slot(performance(prediction(sort(pred.ann.prob,decreasing=F),test.ann$target[order(pred.svm.prob,decreasing=F)])
                          ,"tpr","rpp"),'x.values'))
y=unlist(slot(performance(prediction(sort(pred.ann.prob,decreasing=F),test.ann$target[order(pred.svm.prob,decreasing=F)])
                          ,"tpr","rpp"),'y.values'))
par(mfrow=c(1,1))
par(pty="m")
plot(x,y,type="l",col="green",lwd=4,main="Cumulative gain chart for ANN model",font.main=2,xlab="Rate of Positive Prediction",
     ylab="Cumulative proportion of target")
abline(a=0,b=1,col="red")
text(0.8,0.2,"Area under model\n curve is 0.7756",font = 2,col="steelblue4")
text(0.1,0.85,"best line->")
text(0.8,0.69,"<-base line")
lines(x=c(0,0.2166667,1),y=c(0,1,1),lwd=2,col="blue")