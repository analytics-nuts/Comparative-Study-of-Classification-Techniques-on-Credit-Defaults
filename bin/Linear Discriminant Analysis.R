##Fitting a Linear discriminent model ##

model.lda=lda(target~.,data=train.logit,prior=c(0.7788,0.2212))
model.lda

#For test set
pred.lda=predict(model.lda,test.logit)
head(pred.lda$posterior)
data.frame(pred.lda$posterior)
pred.lda.prob=pred.lda$posterior[,2]

table(predict=pred.lda$class,true=test.logit$target)
err2[2]=(840+230)/6000

#For train set
pred.lda.train=predict(model.lda,train.logit)
data.frame(pred.lda.train$posterior)

table(predict=pred.lda.train$class,true=train.logit$target)
err1[2]=(3294+991)/24000

##Ploting ROC curve and AUC for test and train set

par(mfrow=c(1,2))
par(pty="s")

roc(train.logit$target,pred.lda.train$posterior[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")

roc(test.logit$target,pred.lda.prob,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc1[2]=0.770
auc2[2]=0.768

#Cumulative gain chart for test set
area_mod=performance(prediction(sort(pred.lda.prob,decreasing=F),test.logit$target[order(pred.lda.prob,decreasing=F)])
                     ,measure = "auc")@y.values[[1]] #Area under curve

x=unlist(slot(performance(prediction(sort(pred.lda.prob,decreasing=F),test.logit$target[order(pred.lda.prob,decreasing=F)])
                          ,"tpr","rpp"),'x.values'))
y=unlist(slot(performance(prediction(sort(pred.lda.prob,decreasing=F),test.logit$target[order(pred.lda.prob,decreasing=F)])
                          ,"tpr","rpp"),'y.values'))
par(mfrow=c(1,1))
par(pty="m")
plot(x,y,type="l",col="green",lwd=4,main="Cumulative gain chart for LDA model",font.main=2,xlab="Rate of Positive Prediction",
     ylab="Cumulative proportion of target")
abline(a=0,b=1,col="red")
text(0.8,0.2,"Area under model\n curve is 0.7677",font = 2,col="steelblue4")
text(0.1,0.85,"best line->")
text(0.8,0.69,"<-base line")
lines(x=c(0,0.2166667,1),y=c(0,1,1),lwd=2,col="blue")