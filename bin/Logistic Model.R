## Logistic Model ##
#Splitting the into test and train sets in 80:20 ratio

set.seed(666)#for reproducability of result
ind=sample(nrow(all.features),24000,replace = F)

train.logit=all.features[ind,]
test.logit=all.features[-ind,]

##Fitting a logistic model

model.logit=glm(target~.,data=train.logit,family="binomial")

summary(model.logit)

#For test set
pred.logit=predict(model.logit,type="response",newdata = test.logit)

pred.def=ifelse(pred.logit>0.5,"1","0")

table(predict=pred.def,true=test.logit$target)
err2[1]=(871+209)/6000

#For training set
pred.def=ifelse(predict(model.logit,type="response",newdata = train.logit)>0.5,"1","0")

table(predict=pred.def,true=train.logit$target)
err1[1]=(3395+874)/24000

##Ploting ROC curve and AUC for test and train set

par(mfrow=c(1,2))
par(pty="s")

#For training set
roc(train.logit$target,model.logit$fitted.values,plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")

#For test set
roc(test.logit$target,pred.logit,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc1[1]=0.772
auc2[1]=0.769

#Cumulative gain chart for test set
area_mod=performance(prediction(sort(pred.logit,decreasing=F),test.logit$target[order(pred.logit,decreasing=F)])
                     ,measure = "auc")@y.values[[1]] #Area under curve

x=unlist(slot(performance(prediction(sort(pred.logit,decreasing=F),test.logit$target[order(pred.logit,decreasing=F)])
                          ,"tpr","rpp"),'x.values'))
y=unlist(slot(performance(prediction(sort(pred.logit,decreasing=F),test.logit$target[order(pred.logit,decreasing=F)])
                          ,"tpr","rpp"),'y.values'))
par(mfrow=c(1,1))
par(pty="m")
plot(x,y,type="l",col="green",lwd=4,main="Cumulative gain chart for Logit model",font.main=2,xlab="Rate of Positive Prediction",
     ylab="Cumulative proportion of target")
abline(a=0,b=1,col="red")
text(0.8,0.2,"Area under model\n curve is 0.7688",font = 2,col="steelblue4")
text(0.1,0.85,"best line->")
text(0.8,0.69,"<-base line")
lines(x=c(0,0.2166667,1),y=c(0,1,1),lwd=2,col="blue")